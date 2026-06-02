import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import re
import os

USE_REAL_CSI = True
SHOW_PLOT = True
CSI_CSV_PATH = r"./4.5m_L=4.5m_2.csv"
OUTPUT_DIR = r"./music_spectrums"

SUBARRAY_SIZE = 80
MULTIPATH_L = 3

SNR_DB = 30
np.random.seed(42)

PATH_CONFIGS = [
    (0.0, 0, 0),
    (-2.0, 45, 30),
    (-6.0, 45.8, 60)
]

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


def fast_eigenvalue_analysis(csi_processed, csi_indices, m_range_coarse, verbose=True):
    eigenvalue_ratios = []

    if verbose:
        print("  [阶段1] 快速特征值分析...")

    for m in m_range_coarse:
        N = len(csi_processed)
        m_actual = min(m, N - 10)
        k_val = N - m_actual + 1

        if k_val <= 0:
            m_actual = N // 2
            k_val = N - m_actual + 1

        J_m = np.fliplr(np.eye(m_actual, dtype=np.complex128))
        X_fwd = np.array([csi_processed[i:i + m_actual] for i in range(k_val)])
        X_bwd = (X_fwd.conj() @ J_m)
        X_comb = np.concatenate([X_fwd, X_bwd], axis=0)
        R = (X_comb.conj().T @ X_comb) / (2 * k_val)

        eigvals = np.linalg.eigvalsh(R)
        eigvals = np.abs(eigvals[::-1])
        eigenvalue_ratio = eigvals[0] / (eigvals[1] + 1e-10)
        eigenvalue_ratios.append(eigenvalue_ratio)

        if verbose:
            print(f"    m={m:3d}: 特征值比={eigenvalue_ratio:6.2f}")

    return eigenvalue_ratios


def select_m_range_by_eigenvalue(eigenvalue_ratios, m_range_coarse, verbose=True):
    if verbose:
        print("  [分析] 特征值比趋势...")

    diffs = np.diff(eigenvalue_ratios)
    stable_threshold = 0.5
    stable_indices = np.where(np.abs(diffs) < stable_threshold)[0]

    if len(stable_indices) > 0:
        first_stable_idx = stable_indices[0]
        optimal_m_coarse = m_range_coarse[first_stable_idx]
        if verbose:
            print(f"    特征值比在 m={optimal_m_coarse} 后趋于稳定")
    else:
        optimal_m_coarse = m_range_coarse[len(m_range_coarse) // 2]
        if verbose:
            print(f"    特征值比持续变化，选择中间值 m={optimal_m_coarse}")

    if optimal_m_coarse <= 40:
        search_range = [40, 50, 60]
    elif optimal_m_coarse <= 60:
        search_range = [50, 60, 70]
    elif optimal_m_coarse <= 80:
        search_range = [60, 70, 80]
    else:
        search_range = [70, 80, 90]

    if verbose:
        print(f"    缩小搜索范围: {search_range}")

    return search_range


def fine_scan_single_frame(csi_processed, csi_indices, search_range, L, verbose=True):
    if verbose:
        print("  [阶段2] 单帧精细扫描...")

    results = []

    for m in search_range:
        spectrum, delay_range, eigvals, L_used, R, sc_indices = music_delay_single_frame(
            csi_processed, csi_indices, DELTA_F,
            L=L,
            subarray_size=m
        )

        peak_taus, peak_powers, peak_powers_db = find_music_peaks(
            spectrum, delay_range
        )

        num_valid_peaks = 0
        if len(peak_taus) > 0:
            peak_taus_flipped = -peak_taus

            powers_linear = estimate_joint_path_power(
                csi_processed, csi_indices, DELTA_F, peak_taus_flipped
            )

            idx_first = np.argmin(peak_taus_flipped)
            power_first = powers_linear[idx_first]
            power_threshold = power_first / 10.0
            valid_mask = powers_linear >= power_threshold
            num_valid_peaks = np.sum(valid_mask)

        results.append({
            'm': m,
            'num_peaks': len(peak_taus),
            'num_valid_peaks': num_valid_peaks,
            'eigenvalue_ratio': eigvals[0] / (eigvals[1] + 1e-10)
        })

        if verbose:
            print(f"    m={m:3d}: 检测到{len(peak_taus)}个峰值, 有效{num_valid_peaks}个")

    return results


def select_optimal_m(results, verbose=True):
    if verbose:
        print("  [选择] 最优m值...")

    max_valid_peaks = max([r['num_valid_peaks'] for r in results])
    candidates = [r for r in results if r['num_valid_peaks'] == max_valid_peaks]

    if len(candidates) == 1:
        optimal = candidates[0]
    else:
        optimal = min(candidates, key=lambda x: x['m'])

    if verbose:
        print(f"    最优m值: {optimal['m']}")
        print(f"    有效多径数: {optimal['num_valid_peaks']}")

    return optimal['m']


def adaptive_m_selection(csi_processed, csi_indices, verbose=True):
    if verbose:
        print("\n" + "=" * 60)
        print("自适应m值选择")
        print("=" * 60)

    m_range_coarse = [24, 40, 60, 80, 100]

    eigenvalue_ratios = fast_eigenvalue_analysis(
        csi_processed, csi_indices, m_range_coarse, verbose=verbose
    )

    search_range = select_m_range_by_eigenvalue(
        eigenvalue_ratios, m_range_coarse, verbose=verbose
    )

    results = fine_scan_single_frame(
        csi_processed, csi_indices, search_range, MULTIPATH_L, verbose=verbose
    )

    optimal_m = select_optimal_m(results, verbose=verbose)

    if verbose:
        print("=" * 60)

    return optimal_m


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


def plot_music_spectrum(delay_range, spectrum, peak_taus_original, title="MUSIC Pseudospectrum"):
    plt.figure(figsize=(10, 5))
    plt.plot(delay_range * 1e9, 10 * np.log10(spectrum + 1e-10))
    plt.xlabel("Delay (ns)")
    plt.ylabel("MUSIC Spectrum (dB)")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if len(peak_taus_original) > 0:
        for tau in peak_taus_original:
            idx = np.argmin(np.abs(delay_range - tau))
            plt.plot(tau * 1e9, 10 * np.log10(spectrum[idx] + 1e-10), 'rx', markersize=8)
            plt.text(tau * 1e9, 10 * np.log10(spectrum[idx] + 1e-10) + 2, f'{tau * 1e9:.1f}ns', ha='center')

    plt.tight_layout()

    if SHOW_PLOT:
        plt.show()
    else:
        plt.savefig('music_spectrum.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("谱图已保存为 music_spectrum.png")


def plot_4music_spectrums(results_batch, output_dir, batch_num):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, r in enumerate(results_batch):
        ax = axes[i]
        ax.plot(r['delay_range'] * 1e9, 10 * np.log10(r['spectrum'] + 1e-10))
        ax.set_xlabel("Delay (ns)")
        ax.set_ylabel("MUSIC Spectrum (dB)")
        ax.set_title(f"Frame {r['frame']}")
        ax.grid(True, alpha=0.3)

        if len(r['peak_taus_original']) > 0:
            for tau in r['peak_taus_original']:
                idx = np.argmin(np.abs(r['delay_range'] - tau))
                ax.plot(tau * 1e9, 10 * np.log10(r['spectrum'][idx] + 1e-10), 'rx', markersize=8)
                ax.text(tau * 1e9, 10 * np.log10(r['spectrum'][idx] + 1e-10) + 2,
                        f'{tau * 1e9:.1f}ns', ha='center', fontsize=8)

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f'music_spectrum_batch_{batch_num}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"谱图已保存: {output_path}")


if __name__ == '__main__':
    if USE_REAL_CSI:
        print("=" * 60)
        print("使用真实CSI数据")
        print("=" * 60)

        print(f"从CSV文件读取: {CSI_CSV_PATH}")
        csi_raw_list, rtt_list, frame_numbers = load_csi_from_csv(CSI_CSV_PATH)
        print(f"读取到 {len(csi_raw_list)} 帧CSI数据")

        if len(csi_raw_list) == 0:
            print("错误: 未读取到有效的CSI数据")
            exit(1)

        csi_processed_list = []
        for idx, csi_raw in enumerate(csi_raw_list):
            csi_complex = csi_array_to_complex(csi_raw)
            csi_processed, csi_indices = process_real_csi(csi_complex, DELTA_F)
            csi_processed_list.append(csi_processed)

        print(f"\n成功处理 {len(csi_processed_list)} 帧CSI数据")
        print(f"每帧CSI长度: {len(csi_processed_list[0])} 个复数")
        print(f"子载波索引范围: {csi_indices[0]} 到 {csi_indices[-1]}")

        if SUBARRAY_SIZE is None:
            print("\n使用自适应m值选择...")
            optimal_m = adaptive_m_selection(
                csi_processed_list[0], csi_indices, verbose=True
            )
            subarray_size_to_use = optimal_m
        else:
            subarray_size_to_use = SUBARRAY_SIZE
            print(f"\n使用手动设置m值: {subarray_size_to_use}")

        print("\n处理CSI帧...")
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

        print(f"\n处理完成，共 {len(results)} 帧")

        results = filter_by_length_mad(results, verbose=True)

        print("\n" + "=" * 100)
        print("CSI帧时延估计结果（过滤后）")
        print("=" * 100)
        print(
            f"{'帧号':<6} {'RTT(ns)':<10} {'修正时延(ns)':<14} {'RTT/2(ns)':<11} {'长度(m)':<10} {'多径数':<8} {'备注'}")
        print("-" * 100)

        for idx, r in enumerate(results):
            if r['is_valid']:
                note = ""
                if r['num_peaks'] > 3:
                    note = "多径复杂"

                print(
                    f"{r['frame']:<6} {r['rtt_ns']:<10} {r['correction_delay_ns']:<14.2f} {r['rtt_half_ns']:<11.2f} {r['length_m']:<10.2f} {r['num_peaks']:<8} {note}")

                if len(r['delays_ns']) > 0:
                    print(f"        多径详情: ", end="")
                    for j, (delay, power_db, power_lin) in enumerate(
                            zip(r['delays_ns'], r['powers_db'], r['powers_linear'])):
                        print(f"τ{delay:.1f}ns({power_db:.1f}dB,{power_lin:.3f}) ", end="")
                    print()
            else:
                print(
                    f"{r['frame']:<6} {r['rtt_ns']:<10} {'N/A':<14} {r['rtt_half_ns']:<11.2f} {'N/A':<10} {'0':<8} 无峰值")

        print("=" * 100)

        if len(results) > 0:
            print(f"\n生成MUSIC谱图到文件夹: {OUTPUT_DIR}")
            for i in range(0, len(results), 4):
                batch = results[i:i + 4]
                batch_num = i // 4 + 1
                plot_4music_spectrums(batch, OUTPUT_DIR, batch_num)

        if len(results) > 0:
            print("\n" + "=" * 100)
            print("统计信息")
            print("=" * 100)

            total_frames = len(results)
            valid_frames = [r for r in results if r['is_valid']]
            invalid_frames = [r for r in results if not r['is_valid']]

            print(f"总帧数: {total_frames}")
            print(f"有效帧数: {len(valid_frames)} (多径数 >= 2)")
            print(f"无效帧数: {len(invalid_frames)} (多径数 < 2)")

            if len(valid_frames) > 0:
                lengths = [r['length_m'] for r in valid_frames if r['length_m'] is not None]
                if len(lengths) > 0:
                    print(f"\n有效帧长度统计:")
                    print(f"  平均长度: {np.mean(lengths):.2f} m")
                    print(f"  标准差: {np.std(lengths):.2f} m")
                    print(f"  最小长度: {np.min(lengths):.2f} m")
                    print(f"  最大长度: {np.max(lengths):.2f} m")

                print("\n" + "=" * 100)
                print("详细多径信息（前5个有效帧）")
                print("=" * 100)
                for i, r in enumerate(valid_frames[:5]):
                    print(f"\n帧 {r['frame']}:")
                    print(f"  RTT: {r['rtt_ns']} ns")
                    print(f"  修正时延: {r['correction_delay_ns']:.2f} ns")
                    print(f"  长度: {r['length_m']:.2f} m")
                    print(f"  检测到 {r['num_peaks_detected']} 个多径，有效 {r['num_peaks']} 个多径:")
                    for j, (delay_orig, delay_flip, power_db, power_lin) in enumerate(
                            zip(r['delays_original_ns'], r['delays_ns'], r['powers_db'], r['powers_linear'])):
                        print(
                            f"    路径{j + 1}: 原始={delay_orig:.2f} ns, 翻转={delay_flip:.2f} ns, 功率={power_db:.2f} dB (线性: {power_lin:.6f})")

    else:
        print("=" * 60)
        print("使用模拟CSI数据")
        print("=" * 60)

        PATHS = [path_from_dB_and_phase(*cfg) for cfg in PATH_CONFIGS]
        subcarrier_indices_all = np.arange(-64, 64)
        f_all = subcarrier_indices_all * DELTA_F

        mask = np.ones(128, dtype=bool)
        mask[:6] = False
        mask[63:66] = False
        mask[123:] = False

        valid_freqs = f_all[mask]
        valid_indices = np.concatenate([np.arange(-58, -1), np.arange(2, 59)])

        H_full = np.zeros(128, dtype=np.complex128)
        for alpha, tau in PATHS:
            H_full += alpha * np.exp(-1j * 2 * np.pi * f_all * tau)

        h_valid = H_full[mask]

        signal_pwr = np.mean(np.abs(h_valid) ** 2)
        noise_pwr = signal_pwr / (10 ** (SNR_DB / 10))
        h_noisy = h_valid + np.sqrt(noise_pwr / 2) * (np.random.randn(114) + 1j * np.random.randn(114))

        target_indices = np.arange(-58, 59)
        target_freqs = target_indices * DELTA_F

        f_interp_real = interp1d(valid_freqs, h_noisy.real, kind='linear', fill_value='extrapolate')
        f_interp_imag = interp1d(valid_freqs, h_noisy.imag, kind='linear', fill_value='extrapolate')
        h_interp = f_interp_real(target_freqs) + 1j * f_interp_imag(target_freqs)

        csi_processed = h_interp
        csi_indices = target_indices

        print(f"CSI数据长度: {len(csi_processed)} 个复数")
        print(f"子载波索引范围: {csi_indices[0]} 到 {csi_indices[-1]}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        phase_raw = np.angle(h_valid)
        phase_interp = np.angle(csi_processed)
        amp_raw = np.abs(h_valid)
        amp_interp = np.abs(csi_processed)

        axes[0, 0].plot(valid_indices, phase_raw, 'b-', linewidth=2, label='Raw Phase')
        axes[0, 0].set_title('Simulated CSI: Raw Phase (114 points)')
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('Phase (rad)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(target_indices, phase_interp, 'r-', linewidth=2, label='Interpolated Phase')
        axes[0, 1].scatter(valid_indices, phase_raw, color='blue', s=30, alpha=0.7, label='Original Points')
        axes[0, 1].set_title('Simulated CSI: Interpolated Phase (117 points)')
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('Phase (rad)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        axes[1, 0].plot(valid_indices, amp_raw, 'b-', linewidth=2, label='Raw Amplitude')
        axes[1, 0].set_title('Simulated CSI: Raw Amplitude (114 points)')
        axes[1, 0].set_xlabel('Subcarrier Index')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].plot(target_indices, amp_interp, 'r-', linewidth=2, label='Interpolated Amplitude')
        axes[1, 1].scatter(valid_indices, amp_raw, color='blue', s=30, alpha=0.7, label='Original Points')
        axes[1, 1].set_title('Simulated CSI: Interpolated Amplitude (117 points)')
        axes[1, 1].set_xlabel('Subcarrier Index')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('simulated_csi_phase.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("模拟CSI相位图已保存: simulated_csi_phase.png")

        if SUBARRAY_SIZE is None:
            print("\n使用自适应m值选择...")
            subarray_size_to_use = adaptive_m_selection(
                csi_processed, csi_indices, verbose=True
            )
        else:
            subarray_size_to_use = SUBARRAY_SIZE
            print(f"\n使用手动设置m值: {subarray_size_to_use}")

        print("\n运行单帧MUSIC算法...")
        spectrum, delay_range, eigvals, L_used, R, sc_indices_used = music_delay_single_frame(
            csi_processed, csi_indices, DELTA_F,
            L=MULTIPATH_L,
            subarray_size=subarray_size_to_use
        )

        print("\n" + "=" * 60)
        print("MUSIC算法结果")
        print("=" * 60)
        print(f"协方差矩阵大小: {sc_indices_used.shape[0]}")
        print(f"指定的多径数 L: {MULTIPATH_L} ，实际使用 L: {L_used}")
        print(f"特征值（前6个）: {eigvals[:6]}")

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

            print("\n估计的时延及功率（联合最小二乘）：")
            for tau, p_db, p_lin in zip(peak_taus, powers_db, powers_linear):
                print(f"  时延: {tau * 1e9:.2f} ns, 功率: {p_db:.2f} dB (线性: {p_lin:.6f})")

            idx_first = np.argmin(peak_taus)
            tau_first = peak_taus[idx_first]
            power_first = powers_linear[idx_first]

            power_threshold = power_first / 10.0
            valid_mask = powers_linear >= power_threshold

            peak_taus_valid = peak_taus[valid_mask]
            powers_linear_valid = powers_linear[valid_mask]
            powers_db_valid = powers_db[valid_mask]

            peak_taus_original_valid = peak_taus_original[valid_mask]

            print(f"\n功率过滤：检测到 {len(peak_taus)} 个多径，有效 {len(peak_taus_valid)} 个多径")

            if len(peak_taus_valid) > 0:
                idx_first_valid = np.argmin(peak_taus_valid)
                tau_first_valid = peak_taus_valid[idx_first_valid]

                total_power = np.sum(powers_linear_valid)
                if total_power > 0:
                    weights = powers_linear_valid / total_power
                    weighted_diff = np.sum(weights * (peak_taus_valid - tau_first_valid))
                    print(f"\n加权时延差（以首径 {tau_first_valid * 1e9:.2f} ns 为参考）: {weighted_diff * 1e9:.2f} ns")

            plot_music_spectrum(delay_range, spectrum, peak_taus_original_valid,
                                title=f"MUSIC Pseudospectrum (Simulated Data)")
        else:
            print("未检测到有效峰值")