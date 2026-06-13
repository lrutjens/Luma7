#include "app_config.h"

#include <string.h>

#include "nvs.h"
#include "nvs_flash.h"

#define LUMA7_NAMESPACE "luma7"

esp_err_t luma7_config_load(luma7_config_t *cfg)
{
    if (!cfg) {
        return ESP_ERR_INVALID_ARG;
    }
    memset(cfg, 0, sizeof(*cfg));
    strncpy(cfg->server_url, "http://192.168.1.100:8080", sizeof(cfg->server_url) - 1);
    strncpy(cfg->auth_token, "changeme-generate-a-32-char-token-00", sizeof(cfg->auth_token) - 1);
    cfg->touch_threshold = 28000;

    nvs_handle_t handle;
    esp_err_t err = nvs_open(LUMA7_NAMESPACE, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        return err;
    }

    size_t len = sizeof(cfg->wifi_ssid);
    nvs_get_str(handle, LUMA7_WIFI_SSID_KEY, cfg->wifi_ssid, &len);
    len = sizeof(cfg->wifi_password);
    nvs_get_str(handle, LUMA7_WIFI_PASS_KEY, cfg->wifi_password, &len);
    len = sizeof(cfg->server_url);
    nvs_get_str(handle, LUMA7_SERVER_URL_KEY, cfg->server_url, &len);
    len = sizeof(cfg->auth_token);
    nvs_get_str(handle, LUMA7_AUTH_TOKEN_KEY, cfg->auth_token, &len);
    nvs_get_u32(handle, LUMA7_TOUCH_THRESH_KEY, &cfg->touch_threshold);
    nvs_close(handle);
    return ESP_OK;
}

esp_err_t luma7_config_save(const luma7_config_t *cfg)
{
    if (!cfg) {
        return ESP_ERR_INVALID_ARG;
    }
    nvs_handle_t handle;
    esp_err_t err = nvs_open(LUMA7_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }
    nvs_set_str(handle, LUMA7_WIFI_SSID_KEY, cfg->wifi_ssid);
    nvs_set_str(handle, LUMA7_WIFI_PASS_KEY, cfg->wifi_password);
    nvs_set_str(handle, LUMA7_SERVER_URL_KEY, cfg->server_url);
    nvs_set_str(handle, LUMA7_AUTH_TOKEN_KEY, cfg->auth_token);
    nvs_set_u32(handle, LUMA7_TOUCH_THRESH_KEY, cfg->touch_threshold);
    err = nvs_commit(handle);
    nvs_close(handle);
    return err;
}
