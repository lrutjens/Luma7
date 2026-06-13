#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

esp_err_t luma7_http_post_query(
    const char *server_url,
    const char *auth_token,
    const uint8_t *jpeg,
    size_t jpeg_len,
    const uint8_t *wav,
    size_t wav_len,
    char *session_id_out,
    size_t session_id_len);

esp_err_t luma7_http_post_stop(const char *server_url, const char *auth_token, const char *session_id);
