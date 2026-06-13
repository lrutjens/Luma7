#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

esp_err_t luma7_camera_init(void);
esp_err_t luma7_camera_capture_jpeg(uint8_t **jpeg_out, size_t *jpeg_len);
void luma7_camera_power_down(void);
