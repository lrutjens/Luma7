#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#define LUMA7_WIFI_SSID_KEY "wifi_ssid"
#define LUMA7_WIFI_PASS_KEY "wifi_pass"
#define LUMA7_SERVER_URL_KEY "server_url"
#define LUMA7_AUTH_TOKEN_KEY "auth_token"
#define LUMA7_TOUCH_THRESH_KEY "touch_thr"

typedef struct {
    char wifi_ssid[32];
    char wifi_password[64];
    char server_url[128];
    char auth_token[33];
    uint32_t touch_threshold;
} luma7_config_t;

esp_err_t luma7_config_load(luma7_config_t *cfg);
esp_err_t luma7_config_save(const luma7_config_t *cfg);
