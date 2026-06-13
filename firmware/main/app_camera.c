#include "app_camera.h"

#include <stdlib.h>
#include <string.h>

#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "camera";
static bool s_camera_on;

static camera_config_t camera_config(void)
{
    camera_config_t config = {
        .pin_pwdn = -1,
        .pin_reset = -1,
        .pin_xclk = GPIO_NUM_10,
        .pin_sscb_sda = GPIO_NUM_40,
        .pin_sscb_scl = GPIO_NUM_39,
        .pin_d7 = GPIO_NUM_48,
        .pin_d6 = GPIO_NUM_11,
        .pin_d5 = GPIO_NUM_12,
        .pin_d4 = GPIO_NUM_14,
        .pin_d3 = GPIO_NUM_16,
        .pin_d2 = GPIO_NUM_18,
        .pin_d1 = GPIO_NUM_17,
        .pin_d0 = GPIO_NUM_15,
        .pin_vsync = GPIO_NUM_38,
        .pin_href = GPIO_NUM_47,
        .pin_pclk = GPIO_NUM_13,
        .xclk_freq_hz = 20000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_SVGA,
        .jpeg_quality = 12,
        .fb_count = 1,
        .fb_location = CAMERA_FB_IN_PSRAM,
        .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
    };
    return config;
}

esp_err_t luma7_camera_init(void)
{
    esp_err_t err = esp_camera_init(&camera_config());
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "camera init failed: %s", esp_err_to_name(err));
        return err;
    }
    luma7_camera_power_down();
    ESP_LOGI(TAG, "camera ready");
    return ESP_OK;
}

esp_err_t luma7_camera_capture_jpeg(uint8_t **jpeg_out, size_t *jpeg_len)
{
    if (!jpeg_out || !jpeg_len) {
        return ESP_ERR_INVALID_ARG;
    }
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor) {
        sensor->set_reg(sensor, 0x3008, 0xff, 0x00);
    }
    s_camera_on = true;
    vTaskDelay(pdMS_TO_TICKS(120));

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        return ESP_FAIL;
    }
    uint8_t *copy = heap_caps_malloc(fb->len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!copy) {
        esp_camera_fb_return(fb);
        return ESP_ERR_NO_MEM;
    }
    memcpy(copy, fb->buf, fb->len);
    *jpeg_out = copy;
    *jpeg_len = fb->len;
    esp_camera_fb_return(fb);
    luma7_camera_power_down();
    return ESP_OK;
}

void luma7_camera_power_down(void)
{
    sensor_t *sensor = esp_camera_sensor_get();
    if (sensor) {
        sensor->set_reg(sensor, 0x3008, 0xff, 0x42);
    }
    s_camera_on = false;
}
