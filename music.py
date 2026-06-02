import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import re
import os

CSI_CSV_PATH = r"./2.76m_L=4.5m_3.csv"

SUBARRAY_SIZE = 50
MULTIPATH_L = 3

DELTA_F = 312.5e3


def load_csi_from_csv(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern_csi2 = r'csi_data_2 = ""\[(.*?)\]""'
    pattern_csi4 = r'csi_data_4 = ""\[(.*?)\]""'

    matches_csi2 = re.findall(pattern_csi2, content)
    matches_csi4 = re.findall(pattern_csi4, content)

    if len(matches_csi2) > 0 and len(matches_csi4) > 0:
        csi_list = []
        max_frames = min(len(matches_csi2), len(matches_csi4))

        for idx in range(max_frames):
            try:
                data_str = '[' + matches_csi4[idx] + ']'
                data = eval(data_str)
                if len(data) == 256:
                    csi_list.append(data)
            except:
                pass
        return csi_list, [], []

    pattern_new = r'\[#(\d+)\] FTM: RTT=(\d+)ns.*?\| CSI: data=\[(.*?)\]'
    matches_new = re.findall(pattern_new, content)

    csi_list = []
    rtt_list = []
    frame_numbers = []

    for idx in range(len(matches_new)):
        try:
            frame_num = int(matches_new[idx][0])
            rtt_ps = int(matches_new[idx][1])
            rtt_ns = rtt_ps / 1000.0
            data_str = '[' + matches_new[idx][2] + ']'
            data = eval(data_str)
            if len(data) == 256:
                csi_list.append(data)
                rtt_list.append(rtt_ns)
                frame_numbers.append(frame_num)
        except:
            pass

    return csi_list, rtt_list, frame_numbers


def csi_array_to_complex(csi_raw_array):
    csi_complex = []
    for i in range(0, len(csi_raw_array) - 1, 2):
        imag = csi_raw_array[i]
        real = csi_raw_array[i + 1]
        csi_complex.append(complex(real, imag))
    return np.array(csi_complex, dtype=np.complex128)


def calibrate_phase(csi_data, all_ks):
    csi_data = csi_data.copy()

    rotation_factor = np.ones(len(all_ks), dtype=np.complex128)
    rotation_factor[all_ks > 0] = np.exp(-1j * np.pi / 2)
    csi_data = csi_data * rotation_factor

    phase_all = np.angle(csi_data)
    phase_all = np.unwrap(phase_all)

    pilot_k = np.array([-53, -25, -11, 11, 25, 53])
    k_to_idx = {k: idx for idx, k in enumerate(all_ks)}

    pilot_indices = [k_to_idx[k] for k in pilot_k if k in k_to_idx]
    if len(pilot_indices) >= 2:
        pilot_phase = phase_all[pilot_indices]
        pilot_k_valid = pilot_k[[k in k_to_idx for k in pilot_k]]

        A = np.column_stack([pilot_k_valid, np.ones(len(pilot_k_valid))])
        coeffs = np.linalg.lstsq(A, pilot_phase, rcond=None)[0]
        slope, intercept = coeffs
        phase_correction = slope * all_ks + intercept

        phase_all = phase_all - phase_correction

    csi_corrected = np.abs(csi_data) * np.exp(1j * phase_all)

    return csi_corrected


def process_real_csi(csi_complex, delta_f=DELTA_F):
    csi_array = np.array(csi_complex, dtype=np.complex128)

    if len(csi_array) != 128:
        print(f"警告: CSI数据长度为 {len(csi_array)}，期望为 128")

    subcarrier_indices_all = np.arange(-64, 64)
    f_all = subcarrier_indices_all * delta_f

    mask = np.ones(128, dtype=bool)
    mask[:6] = False
    mask[63:66] = False
    mask[123:] = False

    valid_freqs = f_all[mask]
    valid_indices = np.concatenate([np.arange(-58, -1), np.arange(2, 59)])

    if len(csi_array) == 128:
        h_valid = csi_array[mask]
    else:
        h_valid = csi_array
        valid_freqs = np.arange(len(h_valid)) * delta_f

    h_calibrated = calibrate_phase(h_valid, valid_indices)

    target_indices = np.arange(-58, 59)
    target_freqs = target_indices * delta_f

    f_interp_real = interp1d(valid_freqs, h_calibrated.real, kind='linear', fill_value='extrapolate')
    f_interp_imag = interp1d(valid_freqs, h_calibrated.imag, kind='linear', fill_value='extrapolate')
    h_interp = f_interp_real(target_freqs) + 1j * f_interp_imag(target_freqs)
    # ========================================

    return h_interp, target_indices


def path_from_dB_and_phase(power_dB, phase_deg, delay_ns):
    amplitude = 10 ** (power_dB / 20)
    phase_rad = np.deg2rad(phase_deg)
    alpha = amplitude * np.exp(1j * phase_rad)
    tau = delay_ns * 1e-9
    return (alpha, tau)


def build_single_frame_correlation_matrix(csi_single, subcarrier_indices, subarray_size=64):
    N = len(csi_single)
    m = min(subarray_size, N - 10)
    k_val = N - m + 1
    if k_val <= 0:
        m = N // 2
        k_val = N - m + 1

    J_m = np.fliplr(np.eye(m, dtype=np.complex128))
    X_fwd = np.array([csi_single[i:i + m] for i in range(k_val)])
    X_bwd = (X_fwd.conj() @ J_m)
    X_comb = np.concatenate([X_fwd, X_bwd], axis=0)
    R = (X_comb.conj().T @ X_comb) / (2 * k_val)

    sc_indices_used = subcarrier_indices[:m]

    return R, sc_indices_used, m


def music_delay_single_frame(csi_single, subcarrier_indices, freq_delta,
                             L=None, subarray_size=64,
                             delay_min=-100e-9, delay_max=100e-9, delay_steps=10001):
    delay_range = np.linspace(delay_min, delay_max, delay_steps)
    R, sc_indices, m = build_single_frame_correlation_matrix(
        csi_single, subcarrier_indices, subarray_size=subarray_size
    )
    eigvals, U = np.linalg.eigh(R)
    eigvals = np.abs(eigvals[::-1])
    U = U[:, ::-1]

    if L is None:
        L = np.sum(eigvals > np.mean(eigvals) * 5)
        L = max(1, min(L, m - 1))

    En = U[:, L:]

    sv_matrix = np.exp(-1j * 2 * np.pi * sc_indices.reshape(-1, 1) @ (freq_delta * delay_range).reshape(1, -1))
    sv_matrix = sv_matrix / np.sqrt(len(sc_indices))
    projection_sum = np.sum(np.abs(En.conj().T @ sv_matrix) ** 2, axis=0)
    spectrum = 1.0 / (projection_sum + 1e-10)
    spectrum = spectrum / np.max(spectrum)

    return spectrum, delay_range, eigvals, L, R, sc_indices


def filter_by_length_mad(results, verbose=True):
    valid_results = [r for r in results if r['is_valid']]
    if len(valid_results) < 4:
        if verbose:
            print("长度MAD过滤: 有效帧数不足，跳过过滤")
        return results

    length_values = np.array([r['length_m'] for r in valid_results])
    median_length = np.median(length_values)
    mad = np.median(np.abs(length_values - median_length))

    if mad < 1e-6:
        mad = np.std(length_values) * 0.6745

    threshold = 3.0 * mad / 0.6745
    lower_bound = 1.0
    upper_bound = median_length + threshold

    if verbose:
        print(f"\n长度过滤:")
        print(f"  中位数={median_length:.2f}m, MAD={mad:.2f}m")
        print(f"  MAD上限: {upper_bound:.2f}m")
        print(f"  有效范围: [1.00m, {upper_bound:.2f}m]")

    filtered_results = []
    removed_lower = 0
    removed_upper = 0

    for r in results:
        if not r['is_valid']:
            filtered_results.append(r)
        elif r['length_m'] < lower_bound:
            removed_lower += 1
            if verbose:
                print(f"  剔除帧 {r['frame']}: 长度={r['length_m']:.2f}m (< 1.00m)")
        elif r['length_m'] > upper_bound:
            removed_upper += 1
            if verbose:
                print(f"  剔除帧 {r['frame']}: 长度={r['length_m']:.2f}m (> {upper_bound:.2f}m)")
        else:
            filtered_results.append(r)

    if verbose:
        print(f"  长度过滤: 剔除 {removed_lower} 帧(下限), {removed_upper} 帧(上限)")

    return filtered_results


def find_music_peaks(spectrum, delay_range, min_prominence_db=0.5, min_width=1, threshold_db=-40):
    spectrum_db = 10 * np.log10(spectrum + 1e-10)
    peaks, properties = find_peaks(spectrum_db,
                                   prominence=min_prominence_db,
                                   width=min_width)
    if len(peaks) == 0:
        return [], [], []

    max_power = np.max(spectrum_db)
    relative_threshold = max_power + threshold_db
    valid_peaks = [p for p in peaks if spectrum_db[p] >= relative_threshold]
    if len(valid_peaks) == 0:
        return [], [], []

    valid_peaks = np.array(valid_peaks)
    valid_powers = spectrum_db[valid_peaks]

    sorted_indices = np.argsort(valid_powers)[::-1]
    top_peaks = valid_peaks[sorted_indices]
    sorted_by_delay = np.argsort(delay_range[top_peaks])
    top_peaks = top_peaks[sorted_by_delay]

    peak_taus = delay_range[top_peaks]
    peak_powers = spectrum[top_peaks]
    peak_powers_db = spectrum_db[top_peaks]

    return peak_taus, peak_powers, peak_powers_db


def estimate_joint_path_power(csi_data, subcarrier_indices, freq_delta, taus):
    k = subcarrier_indices

    A_cols = []
    for tau in taus:
        a = np.exp(-1j * 2 * np.pi * k * freq_delta * tau)
        A_cols.append(a)

    A = np.column_stack(A_cols)

    h, residuals, rank, s = np.linalg.lstsq(A, csi_data, rcond=None)

    powers = np.abs(h) ** 2

    return powers


if __name__ == '__main__':
    csi_raw_list, rtt_list, frame_numbers = load_csi_from_csv(CSI_CSV_PATH)

    if len(csi_raw_list) == 0:
        print("错误: 未读取到有效的CSI数据")
        exit(1)

    csi_processed_list = []
    for idx, csi_raw in enumerate(csi_raw_list):
        csi_complex = csi_array_to_complex(csi_raw)
        csi_processed, csi_indices = process_real_csi(csi_complex, DELTA_F)
        csi_processed_list.append(csi_processed)

    subarray_size_to_use = SUBARRAY_SIZE

    results = []

    for idx, (csi_processed, rtt_ns, frame_num) in enumerate(zip(csi_processed_list, rtt_list, frame_numbers)):
        spectrum, delay_range, eigvals, L_used, R, sc_indices_used = music_delay_single_frame(
            csi_processed, csi_indices, DELTA_F,
            L=MULTIPATH_L,
            subarray_size=subarray_size_to_use
        )

        peak_taus, peak_powers, peak_powers_db = find_music_peaks(
            spectrum, delay_range
        )

        if len(peak_taus) > 0:
            peak_taus_original = peak_taus.copy()
            peak_taus = -peak_taus

            powers_linear = estimate_joint_path_power(
                csi_processed, csi_indices, DELTA_F, peak_taus
            )
            powers_db = 10 * np.log10(powers_linear + 1e-10)

            idx_first = np.argmin(peak_taus)
            tau_first = peak_taus[idx_first]
            power_first = powers_linear[idx_first]

            power_threshold = power_first / 10.0
            valid_mask = powers_linear >= power_threshold

            peak_taus_valid = peak_taus[valid_mask]
            powers_linear_valid = powers_linear[valid_mask]
            powers_db_valid = powers_db[valid_mask]

            peak_taus_original_valid = peak_taus_original[valid_mask]

            if len(peak_taus_valid) > 0:
                idx_first_valid = np.argmin(peak_taus_valid)
                tau_first_valid = peak_taus_valid[idx_first_valid]

                total_power = np.sum(powers_linear_valid)
                if total_power > 0:
                    weights = powers_linear_valid / total_power
                    correction_delay = np.sum(weights * (peak_taus_valid - tau_first_valid))
                else:
                    correction_delay = 0
            else:
                correction_delay = 0

            correction_delay_ns = correction_delay * 1e9

            rtt_half_ns = rtt_ns / 2.0

            length_m = (rtt_half_ns - correction_delay_ns) * 3e8 / 1e9

            is_valid = len(peak_taus_valid) >= 2

            results.append({
                'frame': idx,
                'rtt_ns': rtt_ns,
                'correction_delay_ns': correction_delay_ns,
                'rtt_half_ns': rtt_half_ns,
                'length_m': length_m,
                'num_peaks': len(peak_taus_valid),
                'num_peaks_detected': len(peak_taus),
                'delays_ns': peak_taus_valid * 1e9,
                'delays_original_ns': peak_taus_original_valid * 1e9,
                'powers_db': powers_db_valid,
                'powers_linear': powers_linear_valid,
                'is_valid': is_valid,
                'spectrum': spectrum,
                'delay_range': delay_range,
                'peak_taus_original': peak_taus_original_valid
            })
        else:
            results.append({
                'frame': idx,
                'rtt_ns': rtt_ns,
                'correction_delay_ns': None,
                'rtt_half_ns': rtt_ns / 2.0,
                'length_m': None,
                'num_peaks': 0,
                'num_peaks_detected': 0,
                'delays_ns': np.array([]),
                'delays_original_ns': np.array([]),
                'powers_db': np.array([]),
                'powers_linear': np.array([]),
                'is_valid': False,
                'spectrum': spectrum,
                'delay_range': delay_range,
                'peak_taus_original': np.array([])
            })

    results = filter_by_length_mad(results, verbose=False)

    print("\n+------+----------+--------------+----------+--------+")
    print("| {:^4} | {:^8} | {:^12} | {:^8} | {:^6} |".format("帧号", "RTT(ns)", "修正时延", "长度", "多径数"))
    print("+------+----------+--------------+----------+--------+")

    for r in results:
        if r['is_valid']:
            print("| {:>4} | {:>8} | {:>12.2f} | {:>8.2f} | {:>6} |".format(
                r['frame'], r['rtt_ns'], r['correction_delay_ns'], r['length_m'], r['num_peaks']))

    print("+------+----------+--------------+----------+--------+")

    if len(results) > 0:
        valid_frames = [r for r in results if r['is_valid']]

        print(f"\n有效帧数: {len(valid_frames)}")

        if len(valid_frames) > 0:
            lengths = [r['length_m'] for r in valid_frames if r['length_m'] is not None]
            if len(lengths) > 0:
                print("\n+---------------------+")
                print("| {:^17} |".format("长度统计"))
                print("+---------------------+")
                print("| 平均: {:>8.2f} m |".format(np.mean(lengths)))
                print("| 标准差: {:>7.2f} m |".format(np.std(lengths)))
                print("| 范围: [{:.2f}, {:.2f}] |".format(np.min(lengths), np.max(lengths)))
                print("+---------------------+")