#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_netif.h"

#define TAG "Initiator"

#define WIFI_CSI_MAXLEN 256
#define WIFI_MODE WIFI_MODE_STA
#define WIFI_INTERFACE WIFI_IF_STA
#define CONFIG_PRIMARY_CHANNEL 11
#define CONFIG_WIFI_2G_BANDWIDTHS   WIFI_BW_HT40
#define CONFIG_WIFI_BAND_MODE       WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_PROTOCOL     WIFI_PROTOCOL_11N

#define IEEE80211_ACTION_CATEGORY_PUBLIC     4
#define IEEE80211_PUBLIC_ACTION_FTM          33

#define CONFIG_GAIN_CONTROL         0
#define CONFIG_FORCE_GAIN           0
#define FTM_FRAME_COUNT             16

#if CONFIG_FORCE_GAIN
extern void phy_fft_scale_force(bool force_en, uint8_t force_value);
extern void phy_force_rx_gain(int force_en, int force_value);
#endif

static const uint8_t CONFIG_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};

#if CONFIG_GAIN_CONTROL

#endif

static SemaphoreHandle_t g_uart_mutex = NULL;
#define SAFE_PRINTF(...) do { \
    if (xSemaphoreTake(g_uart_mutex, portMAX_DELAY) == pdTRUE) { \
        printf(__VA_ARGS__); \
        xSemaphoreGive(g_uart_mutex); \
    } \
} while(0)

static esp_event_handler_instance_t ftm_inst;
static EventGroupHandle_t s_ftm_event_group;
static uint32_t s_rtt_raw, s_rtt_est, s_dist_est;
static uint8_t s_ftm_report_num_entries;
static const int FTM_REPORT_BIT = BIT0;
static const int FTM_FAILURE_BIT = BIT1;

#define MAX_FTM_FRAMES              32  

static portMUX_TYPE g_csi_spinlock = portMUX_INITIALIZER_UNLOCKED;
static SemaphoreHandle_t s_csi_done_sem = NULL;

static uint16_t g_csi_frame_index = 0;
static uint16_t g_csi_lens[MAX_FTM_FRAMES];


typedef struct {
    uint32_t csi_timestamp;
    int8_t csi_rssi;
    uint16_t csi_len;
    int8_t csi_data[WIFI_CSI_MAXLEN];
    uint8_t dialog_token;   
    bool valid;            
#if CONFIG_GAIN_CONTROL
    uint8_t agc_gain;
    uint8_t fft_gain;
#endif
} csi_frame_t;


static csi_frame_t g_csi_frames[MAX_FTM_FRAMES];

static wifi_ftm_report_entry_t g_ftm_report_entries[MAX_FTM_FRAMES];
static bool g_ftm_report_valid = false;

static uint16_t g_measurement_count = 0;

#define MEASUREMENT_COUNT  10

static wifi_ap_record_t *g_ap_list_buffer = NULL;
static uint16_t g_scan_ap_num = 0;
static uint8_t g_target_bssid[6];
static uint8_t g_target_channel;

static void wifi_init(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_INTERFACE, CONFIG_MAC));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(WIFI_INTERFACE, &protocols));
    
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(WIFI_INTERFACE, &bandwidth));
    
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_PRIMARY_CHANNEL, WIFI_SECOND_CHAN_BELOW));
}

static bool is_ftm_action_frame(wifi_csi_info_t *info) {
    if (info->payload_len < 2 || info->payload == NULL) {
        return false;
    }
    
    uint8_t category = info->payload[0];
    uint8_t action = info->payload[1];
    
    return (category == IEEE80211_ACTION_CATEGORY_PUBLIC && 
            action == IEEE80211_PUBLIC_ACTION_FTM);
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) {
        return;
    }
    if (memcmp(info->mac, g_target_bssid, 6) != 0) {
        return;
    }
    if (!is_ftm_action_frame(info)) {
        return;
    }
    
    uint8_t dialog_token = info->payload[2];
    
    portENTER_CRITICAL(&g_csi_spinlock);
    uint8_t idx = g_csi_frame_index;
    if (idx < MAX_FTM_FRAMES) {
        g_csi_frame_index++;
    }
    portEXIT_CRITICAL(&g_csi_spinlock);
    
    if (idx >= MAX_FTM_FRAMES) {
        ESP_LOGW(TAG, "CSI dropped: array full (max %d frames)", MAX_FTM_FRAMES);
        return;
    }
    
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;

#if CONFIG_GAIN_CONTROL
    uint8_t local_agc = rx_ctrl->agc_gain;
    uint8_t local_fft = rx_ctrl->fft_gain;
#endif
    

    g_csi_lens[idx] = (info->len > WIFI_CSI_MAXLEN) ? WIFI_CSI_MAXLEN : info->len;
    g_csi_frames[idx].dialog_token = dialog_token;
    g_csi_frames[idx].valid = true;
    g_csi_frames[idx].csi_timestamp = rx_ctrl->timestamp;
    g_csi_frames[idx].csi_rssi = rx_ctrl->rssi;
    g_csi_frames[idx].csi_len = g_csi_lens[idx];
    memcpy(g_csi_frames[idx].csi_data, info->buf, g_csi_lens[idx]);
#if CONFIG_GAIN_CONTROL
    g_csi_frames[idx].agc_gain = local_agc;
    g_csi_frames[idx].fft_gain = local_fft;
#endif
    
    ESP_LOGI(TAG, "CSI captured: Token=%d, idx=%d, len=%d", dialog_token, idx, g_csi_lens[idx]);
    
    if (g_csi_frame_index >= MAX_FTM_FRAMES && s_csi_done_sem != NULL) {
        xSemaphoreGiveFromISR(s_csi_done_sem, NULL);
    }
}

static void csi_init(void) {
    wifi_csi_config_t csi_config = {
        .enable = true,
        .acquire_csi_legacy = true,
        .acquire_csi_ht20 = false,
        .acquire_csi_ht40 = true,
        .acquire_csi_su = false,
        .acquire_csi_mu = false,
        .acquire_csi_dcm = false,
        .acquire_csi_beamformed = false,
        .val_scale_cfg = false,
        .dump_ack_en = false
    };
    
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    
    s_csi_done_sem = xSemaphoreCreateBinary();
    if (s_csi_done_sem == NULL) {
        ESP_LOGE(TAG, "Failed to create CSI done semaphore");
    }
    
    ESP_LOGI(TAG, "CSI initialized (FTM frame filter + semaphore sync)");
    ESP_LOGI(TAG, "CSI config: legacy=true (L-LTF for FTM frames), ht40=true");
    ESP_LOGI(TAG, "Filter: Category=%d, Action=%d", 
             IEEE80211_ACTION_CATEGORY_PUBLIC, IEEE80211_PUBLIC_ACTION_FTM);
}

static void ftm_event_handler(void *arg, esp_event_base_t event_base,
                              int32_t event_id, void *event_data) {
    if (event_id == WIFI_EVENT_FTM_REPORT) {
        wifi_event_ftm_report_t *event = (wifi_event_ftm_report_t *)event_data;
        s_rtt_raw = event->rtt_raw;
        s_rtt_est = event->rtt_est;
        s_dist_est = event->dist_est;
        s_ftm_report_num_entries = event->ftm_report_num_entries;
        g_ftm_report_valid = false;
        
        if (s_ftm_report_num_entries > 0 && s_ftm_report_num_entries <= FTM_FRAME_COUNT) {
            wifi_ftm_report_entry_t *ftm_report = malloc(sizeof(wifi_ftm_report_entry_t) * s_ftm_report_num_entries);
            if (ftm_report != NULL) {
                memset(ftm_report, 0, sizeof(wifi_ftm_report_entry_t) * s_ftm_report_num_entries);
                esp_err_t ret = esp_wifi_ftm_get_report(ftm_report, s_ftm_report_num_entries);
                if (ret == ESP_OK) {
                    uint16_t copy_count = (s_ftm_report_num_entries <= MAX_FTM_FRAMES) ? 
                                          s_ftm_report_num_entries : MAX_FTM_FRAMES;
                    memcpy(g_ftm_report_entries, ftm_report, sizeof(wifi_ftm_report_entry_t) * copy_count);
                    g_ftm_report_valid = true;
                    ESP_LOGI(TAG, "FTM report saved: entries=%d, rtt_raw=%" PRId32 ", dist=%" PRId32,
                             s_ftm_report_num_entries, s_rtt_raw, s_dist_est / 100);
                } else {
                    ESP_LOGE(TAG, "esp_wifi_ftm_get_report failed: %s", esp_err_to_name(ret));
                }
                free(ftm_report);
            } else {
                ESP_LOGW(TAG, "Failed to allocate memory for FTM report (%d entries)", 
                         s_ftm_report_num_entries);
            }
        } else if (s_ftm_report_num_entries > FTM_FRAME_COUNT) {
            ESP_LOGE(TAG, "Invalid FTM entries count: %d (max: %d), possible hardware anomaly", 
                     s_ftm_report_num_entries, FTM_FRAME_COUNT);
            s_ftm_report_num_entries = 0;
        }
        
        if (event->status == FTM_STATUS_SUCCESS) {
            xEventGroupSetBits(s_ftm_event_group, FTM_REPORT_BIT);
        } else if (event->status == FTM_STATUS_USER_TERM) {
            ESP_LOGI(TAG, "User terminated FTM procedure");
        } else {
            ESP_LOGE(TAG, "FTM failed for peer "MACSTR", status %d", 
                     MAC2STR(event->peer_mac), event->status);
            xEventGroupSetBits(s_ftm_event_group, FTM_FAILURE_BIT);
        }
    }
}

static void ftm_init(void) {
    if (s_ftm_event_group == NULL) {
        s_ftm_event_group = xEventGroupCreate();
        if (!s_ftm_event_group) {
            ESP_LOGE(TAG, "Failed to create FTM event group");
            return;
        }
    }
    
    if (ftm_inst != NULL) {
        esp_event_handler_instance_unregister(WIFI_EVENT, WIFI_EVENT_FTM_REPORT, ftm_inst);
    }
    
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT,
        WIFI_EVENT_FTM_REPORT,
        &ftm_event_handler,
        NULL,
        &ftm_inst
    ));
    
    ESP_LOGI(TAG, "FTM initialized");
}

static bool wifi_perform_scan(void) {
    wifi_scan_config_t scan_config = { 0 };
    scan_config.ssid = NULL;
    scan_config.show_hidden = true;
    
    if (esp_wifi_scan_start(&scan_config, true) != ESP_OK) {
        ESP_LOGE(TAG, "Scan start failed");
        return false;
    }
    
    esp_wifi_scan_get_ap_num(&g_scan_ap_num);
    if (g_scan_ap_num == 0) {
        ESP_LOGI(TAG, "No AP found");
        return false;
    }
    
    if (g_ap_list_buffer) {
        free(g_ap_list_buffer);
        g_ap_list_buffer = NULL;
    }
    
    g_ap_list_buffer = malloc(sizeof(wifi_ap_record_t) * g_scan_ap_num);
    if (!g_ap_list_buffer) {
        ESP_LOGE(TAG, "Malloc failed for AP list");
        esp_wifi_clear_ap_list();
        return false;
    }
    
    if (esp_wifi_scan_get_ap_records(&g_scan_ap_num, g_ap_list_buffer) != ESP_OK) {
        ESP_LOGE(TAG, "Scan get AP records failed");
        free(g_ap_list_buffer);
        g_ap_list_buffer = NULL;
        return false;
    }
    
    ESP_LOGI(TAG, "Scan done, %d APs found", g_scan_ap_num);
    for (int i = 0; i < g_scan_ap_num; ++i) {
        ESP_LOGI(TAG, "  [%d] %s ch=%d rssi=%d ftm=%s", 
                 i, g_ap_list_buffer[i].ssid, 
                 g_ap_list_buffer[i].primary,
                 g_ap_list_buffer[i].rssi,
                 g_ap_list_buffer[i].ftm_responder ? "YES" : "NO");
    }
    
    return true;
}

static bool find_ftm_responder(void) {
    for (int i = 0; i < g_scan_ap_num; ++i) {
        if (g_ap_list_buffer[i].ftm_responder) {
            memcpy(g_target_bssid, g_ap_list_buffer[i].bssid, 6);
            g_target_channel = g_ap_list_buffer[i].primary;
            ESP_LOGI(TAG, "Found FTM Responder: %s ["MACSTR"] ch=%d",
                     g_ap_list_buffer[i].ssid, MAC2STR(g_target_bssid), g_target_channel);
            return true;
        }
    }
    return false;
}

static void print_measurement_result(void) {

    vTaskDelay(pdMS_TO_TICKS(50));
    
    SAFE_PRINTF("\n");
    SAFE_PRINTF("================================================================\n");
    SAFE_PRINTF("第%d组测量完成\n", g_measurement_count);
    SAFE_PRINTF("  平均RTT(raw): %" PRId32 " ps\n", s_rtt_raw);
    SAFE_PRINTF("  平均RTT(est): %" PRId32 " ps\n", s_rtt_est);
    SAFE_PRINTF("  平均距离: %" PRId32 ".%02" PRId32 " m\n", s_dist_est / 100, s_dist_est % 100);
    SAFE_PRINTF("  有效FTM帧: %d / %d\n", s_ftm_report_num_entries, FTM_FRAME_COUNT);
    SAFE_PRINTF("  捕获CSI帧: %d / %d\n", g_csi_frame_index, MAX_FTM_FRAMES);
    SAFE_PRINTF("================================================================\n\n");
    

    if (g_ftm_report_valid && s_ftm_report_num_entries > 0 && g_csi_frame_index > 0) {
        SAFE_PRINTF("---------- FTM-CSI 配对数据 (Dialog Token匹配) ----------\n");
        
        for (int i = 0; i < s_ftm_report_num_entries; i++) {
            uint8_t dialog_token = g_ftm_report_entries[i].dlog_token;
            
   
            int csi_match_idx = -1;
            for (int j = 0; j < g_csi_frame_index; j++) {
                if (g_csi_frames[j].valid && g_csi_frames[j].dialog_token == dialog_token) {
                    csi_match_idx = j;
                    break;
                }
            }

            if (xSemaphoreTake(g_uart_mutex, portMAX_DELAY) == pdTRUE) {
                if (csi_match_idx >= 0) {

                    printf("[#%02d] Token=%d FTM: RTT=%" PRId32 "ps t1=%llu t2=%llu t3=%llu t4=%llu | CSI: RSSI=%d data=[",
                           i + 1,
                           dialog_token,
                           g_ftm_report_entries[i].rtt,
                           g_ftm_report_entries[i].t1,
                           g_ftm_report_entries[i].t2,
                           g_ftm_report_entries[i].t3,
                           g_ftm_report_entries[i].t4,
                           g_csi_frames[csi_match_idx].csi_rssi);
                    
                    for (int j = 0; j < g_csi_frames[csi_match_idx].csi_len; j++) {
                        printf("%d", g_csi_frames[csi_match_idx].csi_data[j]);
                        if (j < g_csi_frames[csi_match_idx].csi_len - 1) {
                            printf(",");
                        }
                        
                        if (j % 32 == 31) {
                            xSemaphoreGive(g_uart_mutex);
                            vTaskDelay(pdMS_TO_TICKS(1));
                            if (xSemaphoreTake(g_uart_mutex, portMAX_DELAY) != pdTRUE) {
                                break;
                            }
                        }
                    }
                    printf("]\n");
                } else {

                    printf("[#%02d] Token=%d FTM: RTT=%" PRId32 "ps t1=%llu t2=%llu t3=%llu t4=%llu | CSI: MISSING\n",
                           i + 1,
                           dialog_token,
                           g_ftm_report_entries[i].rtt,
                           g_ftm_report_entries[i].t1,
                           g_ftm_report_entries[i].t2,
                           g_ftm_report_entries[i].t3,
                           g_ftm_report_entries[i].t4);
                }
                xSemaphoreGive(g_uart_mutex);
            }
        }
        
        SAFE_PRINTF("---------------------------------------\n\n");
    } else {
        SAFE_PRINTF("FTM报告无效或无CSI数据\n\n");
    }
}

static void reset_csi_buffer(void) {
    g_csi_frame_index = 0;
    memset(g_csi_lens, 0, sizeof(g_csi_lens));
    memset(g_csi_frames, 0, sizeof(g_csi_frames)); // Clear CSI cache
    g_ftm_report_valid = false;
    if (s_csi_done_sem != NULL) {
        xSemaphoreTake(s_csi_done_sem, 0);
    }
}

static void perform_single_measurement(void) {
    reset_csi_buffer();
    g_measurement_count++;
    
    SAFE_PRINTF("\n========== 第%d组开始FTM测量 ==========\n", g_measurement_count);
    
    wifi_ftm_initiator_cfg_t ftm_cfg = {
        .frm_count = FTM_FRAME_COUNT,
        .burst_period = 2,
        .use_get_report_api = true
    };
    memcpy(ftm_cfg.resp_mac, g_target_bssid, 6);
    ftm_cfg.channel = g_target_channel;
    
    if (esp_wifi_ftm_initiate_session(&ftm_cfg) != ESP_OK) {
        ESP_LOGE(TAG, "FTM initiate failed");
        return;
    }
    
    EventBits_t bits = xEventGroupWaitBits(s_ftm_event_group,
        FTM_REPORT_BIT | FTM_FAILURE_BIT,
        pdTRUE, pdFALSE, pdMS_TO_TICKS(5000));
    
    if (bits & FTM_REPORT_BIT) {
        if (xSemaphoreTake(s_csi_done_sem, pdMS_TO_TICKS(2000)) == pdTRUE) {
            print_measurement_result();
        } else {
            vTaskDelay(pdMS_TO_TICKS(200));
            if (g_csi_frame_index > 0) {
                ESP_LOGW(TAG, "CSI timeout, but got %d/%d frames", 
                         g_csi_frame_index, MAX_FTM_FRAMES);
                print_measurement_result();
            } else {
                ESP_LOGW(TAG, "FTM success but no CSI from FTM frames");
                SAFE_PRINTF("\nMeasurement #%d: FTM OK, CSI MISSING\n", g_measurement_count);
                SAFE_PRINTF("  RTT: %" PRId32 " ns, Dist: %" PRId32 ".%02" PRId32 " m\n",
                       s_rtt_est, s_dist_est / 100, s_dist_est % 100);
                SAFE_PRINTF("  (No FTM frame CSI captured after 2s wait)\n\n");
            }
        }
    } else {
        ESP_LOGE(TAG, "FTM measurement failed");
        reset_csi_buffer();
    }
}

void app_main(void) {
    g_uart_mutex = xSemaphoreCreateMutex();
    if (g_uart_mutex == NULL) {
        return;
    }

    esp_log_level_set("wifi", ESP_LOG_ERROR);
    
    SAFE_PRINTF("\n\n");
    SAFE_PRINTF("****************************************************************\n");
    SAFE_PRINTF("*           FTM-CSI Synchronous Measurement System - Initiator *\n");
    SAFE_PRINTF("****************************************************************\n\n");
    SAFE_PRINTF("[System Startup] FTM measurement ready\n");
    
    wifi_init();
    csi_init();
    ftm_init();
    
    ESP_LOGI(TAG, "Waiting for FTM Responder...");
    while (!wifi_perform_scan()) {
        ESP_LOGW(TAG, "Scan failed, retry in 3s...");
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
    
    if (!find_ftm_responder()) {
        ESP_LOGE(TAG, "No FTM Responder found!");
        return;
    }
    

    wifi_second_chan_t second_chan = WIFI_SECOND_CHAN_BELOW;
    if (g_target_channel <= 9) {
        second_chan = WIFI_SECOND_CHAN_ABOVE;
    }
    
    ESP_ERROR_CHECK(esp_wifi_set_channel(g_target_channel, second_chan));
    ESP_LOGI(TAG, "Switched to target channel %d (HT40, second chan %s)",
             g_target_channel,
             (second_chan == WIFI_SECOND_CHAN_ABOVE) ? "ABOVE(+4)" : "BELOW(-4)");
    
    ESP_LOGI(TAG, "Starting FTM+CSI measurements (%d groups)...", MEASUREMENT_COUNT);
    SAFE_PRINTF("\n========== FTM Measurement Started ==========\n\n");
    
    for (int i = 0; i < MEASUREMENT_COUNT; i++) {
        perform_single_measurement();
        vTaskDelay(pdMS_TO_TICKS(500));
    }
    
    SAFE_PRINTF("\n\n========== FTM Measurement Completed ==========\n");
    SAFE_PRINTF("  Total Measurements: %d\n", g_measurement_count);
    SAFE_PRINTF("  Total CSI Frames Captured: %d\n", g_csi_frame_index);
    SAFE_PRINTF("===============================================\n\n");
    
    // Clean up resources
    if (g_ap_list_buffer) {
        free(g_ap_list_buffer);
        g_ap_list_buffer = NULL;
    }
    
    if (s_csi_done_sem != NULL) {
        vSemaphoreDelete(s_csi_done_sem);
        s_csi_done_sem = NULL;
    }
    
    if (g_uart_mutex != NULL) {
        vSemaphoreDelete(g_uart_mutex);
        g_uart_mutex = NULL;
    }
    
    ESP_LOGI(TAG, "System stopped. Press reset to restart.");
}