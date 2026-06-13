#include "app_touch.h"

#include "driver/gpio.h"
#include "driver/touch_pad.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "touch";
static const touch_pad_t TOUCH_PAD = TOUCH_PAD_NUM1;
static const gpio_num_t TOUCH_GPIO = GPIO_NUM_1;

static luma7_touch_cb_t s_cb;
static void *s_ctx;
static bool s_pressed;
static uint32_t s_threshold;

static void touch_monitor_task(void *arg)
{
    uint32_t value = 0;
    bool last = false;
    while (true) {
        touch_pad_read_raw_data(TOUCH_PAD, &value);
        bool pressed = value < s_threshold;
        if (pressed != last) {
            last = pressed;
            s_pressed = pressed;
            if (s_cb) {
                s_cb(pressed, s_ctx);
            }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

esp_err_t luma7_touch_init(uint32_t threshold)
{
    s_threshold = threshold;
    ESP_ERROR_CHECK(touch_pad_init());
    ESP_ERROR_CHECK(touch_pad_set_voltage(TOUCH_HVOLT_2V7, TOUCH_LVOLT_0V5, TOUCH_HVOLT_ATTEN_1V));
    ESP_ERROR_CHECK(touch_pad_config(TOUCH_PAD));
    ESP_ERROR_CHECK(touch_pad_filter_start(10));
    xTaskCreatePinnedToCore(touch_monitor_task, "touch", 3072, NULL, 5, NULL, 0);
    ESP_LOGI(TAG, "touch ready on GPIO%d threshold=%u", TOUCH_GPIO, (unsigned)threshold);
    return ESP_OK;
}

void luma7_touch_set_callback(luma7_touch_cb_t cb, void *ctx)
{
    s_cb = cb;
    s_ctx = ctx;
}

bool luma7_touch_is_pressed(void)
{
    return s_pressed;
}
