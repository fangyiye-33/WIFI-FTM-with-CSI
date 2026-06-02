#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <stdlib.h>
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_netif.h"

#define TAG "Responder"
#define WIFI_CSI_MAXLEN 1470
#define WIFI_MODE WIFI_MODE_AP
#define WIFI_INTERFACE WIFI_IF_AP
#define CONFIG_PRIMARY_CHANNEL 11
#define CONFIG_WIFI_2G_BANDWIDTHS   WIFI_BW_HT40
#define CONFIG_WIFI_BAND_MODE       WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_PROTOCOL     WIFI_PROTOCOL_11N

#define IEEE80211_ACTION_CATEGORY_PUBLIC     4
#define IEEE80211_PUBLIC_ACTION_FTM          33

static const uint8_t CONFIG_MAC[] = {0x00,0x1a,0x00,0x00,0x00,0x00};
static const uint8_t PEER_MAC[] = {0x1a,0x00,0x00,0x00,0x00,0x00};

typedef struct {
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
    unsigned : 16;
    unsigned fft_gain : 8;
    unsigned agc_gain : 8;
    unsigned : 32;
    unsigned : 32;
    unsigned : 32;
} wifi_pkt_rx_ctrl_phy_t;

static int16_t g_csi_len = 0;
static int8_t g_csi_data[WIFI_CSI_MAXLEN];
static uint32_t g_csi_timestamp = 0;
static int8_t g_csi_rssi = 0;
static bool g_csi_ready = false;
static uint16_t g_measurement_count = 0;
static uint16_t g_ftm_frame_count = 0;
static uint16_t g_total_frame_count = 0;

static esp_event_handler_instance_t s_ap_event_inst;
static bool s_ap_started = false;

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
    if (info->payload_len < 2) {
        return false;
    }
    
    if (info->payload == NULL) {
        return false;
    }
    
    uint8_t category = info->payload[0];
    uint8_t action = info->payload[1];
    
    if (category == IEEE80211_ACTION_CATEGORY_PUBLIC && 
        action == IEEE80211_PUBLIC_ACTION_FTM) {
        return true;
    }
    
    return false;
}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) {
        return;
    }
    
    g_total_frame_count++;
    
    if (memcmp(info->mac, PEER_MAC, 6) != 0) {
        return;
    }
    
    if (!is_ftm_action_frame(info)) {
        return;
    }
    
    g_ftm_frame_count++;
    g_measurement_count++;
    
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    
    if (info->len > 0 && info->buf) {
        g_csi_len = (info->len > WIFI_CSI_MAXLEN) ? WIFI_CSI_MAXLEN : info->len;
        memcpy(g_csi_data, info->buf, g_csi_len);
        g_csi_timestamp = rx_ctrl->timestamp;
        g_csi_rssi = rx_ctrl->rssi;
        g_csi_ready = true;
        
        printf("\n");
        printf("----------------------------------------------------------------\n");
        printf("FTM帧CSI捕获 [#%d]\n", g_measurement_count);
        printf("----------------------------------------------------------------\n");
        printf("  源MAC: "MACSTR"\n", MAC2STR(info->mac));
        printf("  时间戳: %" PRId32 " us\n", g_csi_timestamp);
        printf("  RSSI: %d dBm\n", g_csi_rssi);
        printf("  CSI长度: %d bytes\n", g_csi_len);
        printf("  帧类型: Category=%d, Action=%d (FTM)\n", 
               IEEE80211_ACTION_CATEGORY_PUBLIC, IEEE80211_PUBLIC_ACTION_FTM);
        printf("  CSI数据: [");
        for (int i = 0; i < g_csi_len && i < 32; i++) {
            printf("%d", g_csi_data[i]);
            if (i < g_csi_len - 1 && i < 31) printf(",");
        }
        if (g_csi_len > 32) printf("...");
        printf("]\n");
        printf("  统计: 总帧数=%d, FTM帧=%d\n", g_total_frame_count, g_ftm_frame_count);
        printf("----------------------------------------------------------------\n\n");
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
    
    ESP_LOGI(TAG, "CSI initialized (FTM frame filter + semaphore sync)");
    ESP_LOGI(TAG, "CSI config: legacy=true (L-LTF for FTM frames), ht40=true");
    ESP_LOGI(TAG, "Filter: Category=%d, Action=%d", 
             IEEE80211_ACTION_CATEGORY_PUBLIC, IEEE80211_PUBLIC_ACTION_FTM);
}

static void ap_event_handler(void *arg, esp_event_base_t event_base,
                             int32_t event_id, void *event_data) {
    if (event_id == WIFI_EVENT_AP_START) {
        s_ap_started = true;
        ESP_LOGI(TAG, "SoftAP started with FTM Responder enabled");
    } else if (event_id == WIFI_EVENT_AP_STOP) {
        s_ap_started = false;
        ESP_LOGI(TAG, "SoftAP stopped");
    } else if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *event = (wifi_event_ap_staconnected_t *)event_data;
        ESP_LOGI(TAG, "Station "MACSTR" connected", MAC2STR(event->mac));
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *event = (wifi_event_ap_stadisconnected_t *)event_data;
        ESP_LOGI(TAG, "Station "MACSTR" disconnected", MAC2STR(event->mac));
    }
}

static esp_err_t start_softap_with_ftm(const char *ssid, const char *pass, uint8_t channel) {
    wifi_config_t ap_config = { 0 };
    strlcpy((char*)ap_config.ap.ssid, ssid, sizeof(ap_config.ap.ssid));
    ap_config.ap.ssid_len = strlen(ssid);
    
    if (pass && strlen(pass) >= 8) {
        strlcpy((char*)ap_config.ap.password, pass, sizeof(ap_config.ap.password));
        ap_config.ap.authmode = WIFI_AUTH_WPA2_PSK;
    } else {
        ap_config.ap.authmode = WIFI_AUTH_OPEN;
    }
    
    ap_config.ap.max_connection = 4;
    ap_config.ap.channel = channel;
    ap_config.ap.ftm_responder = true;
    
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_INTERFACE, &ap_config));
    ESP_LOGI(TAG, "SoftAP configured: SSID=%s, CH=%d, FTM=YES", ssid, channel);
    
    return ESP_OK;
}

void app_main(void) {
    printf("\n\n");
    printf("****************************************************************\n");
    printf("*           FTM-CSI 同步测量系统 - 响应端                      *\n");
    printf("****************************************************************\n\n");
    printf("[系统启动] FTM响应端准备就绪\n");
    
    wifi_init();
    
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, 
        ESP_EVENT_ANY_ID, 
        &ap_event_handler, 
        NULL, 
        &s_ap_event_inst
    ));
    
    csi_init();
    
    const char *ssid = "FTM_RESPONDER";
    const char *pass = "";
    ESP_ERROR_CHECK(start_softap_with_ftm(ssid, pass, CONFIG_PRIMARY_CHANNEL));
    
    vTaskDelay(pdMS_TO_TICKS(500));
    
    if (s_ap_started) {
        ESP_LOGI(TAG, "FTM Responder is running");
        ESP_LOGI(TAG, "AP SSID: %s", ssid);
        ESP_LOGI(TAG, "Channel: %d", CONFIG_PRIMARY_CHANNEL);
        printf("\n========== FTM响应端就绪 ==========\n");
        printf("  SSID: %s\n", ssid);
        printf("  信道: %d\n", CONFIG_PRIMARY_CHANNEL);
        printf("  等待Initiator连接...\n");
        printf("===================================\n\n");
    } else {
        ESP_LOGE(TAG, "Failed to start AP");
        return;
    }
    
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
