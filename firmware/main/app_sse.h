#pragma once

#include "esp_err.h"

typedef void (*luma7_sse_status_cb_t)(const char *state, void *ctx);
typedef void (*luma7_sse_audio_cb_t)(const uint8_t *pcm, size_t len, void *ctx);
typedef void (*luma7_sse_done_cb_t)(void *ctx);
typedef void (*luma7_sse_error_cb_t)(const char *message, void *ctx);

typedef struct {
    luma7_sse_status_cb_t on_status;
    luma7_sse_audio_cb_t on_audio;
    luma7_sse_done_cb_t on_done;
    luma7_sse_error_cb_t on_error;
    void *ctx;
} luma7_sse_callbacks_t;

esp_err_t luma7_sse_stream(
    const char *server_url,
    const char *auth_token,
    const char *session_id,
    const luma7_sse_callbacks_t *callbacks);
