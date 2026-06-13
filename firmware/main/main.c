#include <stdio.h>
#include <string.h>

#include "app_audio.h"
#include "app_camera.h"
#include "app_config.h"
#include "app_http.h"
#include "app_sse.h"
#include "app_touch.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs_flash.h"

static const char *TAG = "main";

#define WIFI_CONNECTED_BIT BIT0

static EventGroupHandle_t s_wifi_event_group;
static luma7_config_t s_cfg;
static char s_active_session[32];
static volatile bool s_playback_active;

typedef enum {
    STATE_IDLE = 0,
    STATE_RECORDING,
    STATE_UPLOADING,
    STATE_STREAMING,
} device_state_t;

static device_state_t s_state = STATE_IDLE;

static void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, s_cfg.wifi_ssid, sizeof(wifi_config.sta.ssid));
    strncpy((char *)wifi_config.sta.password, s_cfg.wifi_password, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    return ESP_OK;
}

static void on_sse_status(const char *state, void *ctx)
{
    ESP_LOGI(TAG, "status: %s", state);
    (void)ctx;
}

static void on_sse_audio(const uint8_t *pcm, size_t len, void *ctx)
{
    (void)ctx;
    s_playback_active = true;
    luma7_audio_playback_push(pcm, len);
}

static void on_sse_done(void *ctx)
{
    (void)ctx;
    s_playback_active = false;
    s_state = STATE_IDLE;
    ESP_LOGI(TAG, "playback complete");
}

static void on_sse_error(const char *message, void *ctx)
{
    (void)ctx;
    ESP_LOGE(TAG, "sse error: %s", message);
    s_playback_active = false;
    s_state = STATE_IDLE;
}

static void query_task(void *arg);

static void start_query_upload(void)
{
    if (s_state != STATE_RECORDING) {
        return;
    }
    s_state = STATE_UPLOADING;
    xTaskCreatePinnedToCore(query_task, "query", 12288, NULL, 5, NULL, 0);
}

static void query_task(void *arg)
{
    uint8_t *jpeg = NULL;
    size_t jpeg_len = 0;
    uint8_t *wav = NULL;
    size_t wav_len = 0;

    if (luma7_camera_capture_jpeg(&jpeg, &jpeg_len) != ESP_OK) {
        jpeg_len = 0;
    }
    wav_len = luma7_audio_stop_record(&wav);
    if (wav_len == 0) {
        free(jpeg);
        s_state = STATE_IDLE;
        vTaskDelete(NULL);
        return;
    }

    if (luma7_http_post_query(
            s_cfg.server_url,
            s_cfg.auth_token,
            jpeg,
            jpeg_len,
            wav,
            wav_len,
            s_active_session,
            sizeof(s_active_session)) != ESP_OK) {
        ESP_LOGE(TAG, "upload failed");
        free(jpeg);
        free(wav);
        s_state = STATE_IDLE;
        vTaskDelete(NULL);
        return;
    }

    free(jpeg);
    free(wav);

    luma7_sse_callbacks_t callbacks = {
        .on_status = on_sse_status,
        .on_audio = on_sse_audio,
        .on_done = on_sse_done,
        .on_error = on_sse_error,
    };
    s_state = STATE_STREAMING;
    luma7_sse_stream(s_cfg.server_url, s_cfg.auth_token, s_active_session, &callbacks);
    vTaskDelete(NULL);
}

static void on_touch(bool pressed, void *ctx)
{
    (void)ctx;
    if (pressed) {
        if (s_state == STATE_STREAMING || s_playback_active) {
            luma7_http_post_stop(s_cfg.server_url, s_cfg.auth_token, s_active_session);
            luma7_audio_playback_stop();
            s_playback_active = false;
            s_state = STATE_IDLE;
            return;
        }
        if (s_state == STATE_IDLE) {
            luma7_audio_start_record();
            s_state = STATE_RECORDING;
        }
        return;
    }

    if (s_state == STATE_RECORDING) {
        start_query_upload();
    }
}

static void device_task(void *arg)
{
    (void)arg;
    while (true) {
        if (s_state == STATE_RECORDING) {
            luma7_audio_poll_record();
            if (luma7_audio_vad_silence_detected()) {
                start_query_upload();
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    luma7_config_load(&s_cfg);

    ESP_ERROR_CHECK(luma7_audio_init());
    ESP_ERROR_CHECK(luma7_camera_init());
    ESP_ERROR_CHECK(luma7_touch_init(s_cfg.touch_threshold));
    luma7_touch_set_callback(on_touch, NULL);

    wifi_init_sta();
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    ESP_LOGI(TAG, "WiFi connected");

    xTaskCreatePinnedToCore(device_task, "device", 4096, NULL, 4, NULL, 0);
    ESP_LOGI(TAG, "Luma7 glasses firmware ready");
}
