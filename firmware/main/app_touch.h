#pragma once

#include <stdbool.h>

#include "esp_err.h"

typedef void (*luma7_touch_cb_t)(bool pressed, void *ctx);

esp_err_t luma7_touch_init(uint32_t threshold);
void luma7_touch_set_callback(luma7_touch_cb_t cb, void *ctx);
bool luma7_touch_is_pressed(void);
