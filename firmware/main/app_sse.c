#include "app_sse.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_http_client.h"
#include "esp_log.h"
#include "mbedtls/base64.h"

static const char *TAG = "sse";

typedef struct {
    char event[32];
    char data[8192];
    size_t data_len;
    const luma7_sse_callbacks_t *callbacks;
} sse_parser_t;

static void dispatch_event(sse_parser_t *parser)
{
    if (!parser->callbacks) {
        return;
    }
    if (strcmp(parser->event, "status") == 0 && parser->callbacks->on_status) {
        char state[32] = "unknown";
        const char *needle = "\"state\":\"";
        char *found = strstr(parser->data, needle);
        if (found) {
            found += strlen(needle);
            char *end = strchr(found, '"');
            if (end) {
                size_t len = (size_t)(end - found);
                if (len < sizeof(state)) {
                    memcpy(state, found, len);
                    state[len] = '\0';
                }
            }
        }
        parser->callbacks->on_status(state, parser->callbacks->ctx);
    } else if (strcmp(parser->event, "audio_chunk") == 0 && parser->callbacks->on_audio) {
        size_t olen = 0;
        uint8_t *decoded = NULL;
        size_t in_len = parser->data_len;
        mbedtls_base64_decode(NULL, 0, &olen, (const unsigned char *)parser->data, in_len);
        decoded = malloc(olen);
        if (decoded && mbedtls_base64_decode(decoded, olen, &olen, (const unsigned char *)parser->data, in_len) == 0) {
            if (olen > 44) {
                parser->callbacks->on_audio(decoded + 44, olen - 44, parser->callbacks->ctx);
            }
        }
        free(decoded);
    } else if (strcmp(parser->event, "audio_done") == 0 && parser->callbacks->on_done) {
        parser->callbacks->on_done(parser->callbacks->ctx);
    } else if (strcmp(parser->event, "error") == 0 && parser->callbacks->on_error) {
        parser->callbacks->on_error(parser->data, parser->callbacks->ctx);
    }
    parser->event[0] = '\0';
    parser->data[0] = '\0';
    parser->data_len = 0;
}

static void feed_line(sse_parser_t *parser, const char *line)
{
    if (strncmp(line, "event:", 6) == 0) {
        const char *value = line + 6;
        while (*value == ' ') {
            value++;
        }
        strncpy(parser->event, value, sizeof(parser->event) - 1);
        return;
    }
    if (strncmp(line, "data:", 5) == 0) {
        const char *value = line + 5;
        while (*value == ' ') {
            value++;
        }
        size_t len = strlen(value);
        if (len >= sizeof(parser->data)) {
            len = sizeof(parser->data) - 1;
        }
        memcpy(parser->data, value, len);
        parser->data[len] = '\0';
        parser->data_len = len;
        return;
    }
    if (line[0] == '\0') {
        if (parser->event[0] != '\0') {
            dispatch_event(parser);
        }
    }
}

static esp_err_t sse_event_handler(esp_http_client_event_t *evt)
{
    sse_parser_t *parser = (sse_parser_t *)evt->user_data;
    if (evt->event_id != HTTP_EVENT_ON_DATA || !parser || evt->data_len <= 0) {
        return ESP_OK;
    }

    static char line[9000];
    static size_t line_len = 0;
    for (int i = 0; i < evt->data_len; i++) {
        char ch = ((char *)evt->data)[i];
        if (ch == '\n') {
            line[line_len] = '\0';
            feed_line(parser, line);
            line_len = 0;
        } else if (ch != '\r') {
            if (line_len + 1 < sizeof(line)) {
                line[line_len++] = ch;
            }
        }
    }
    return ESP_OK;
}

esp_err_t luma7_sse_stream(
    const char *server_url,
    const char *auth_token,
    const char *session_id,
    const luma7_sse_callbacks_t *callbacks)
{
    char url[256];
    snprintf(url, sizeof(url), "%s/stream/%s", server_url, session_id);

    sse_parser_t parser = {0};
    parser.callbacks = callbacks;

    esp_http_client_config_t config = {
        .url = url,
        .method = HTTP_METHOD_GET,
        .event_handler = sse_event_handler,
        .user_data = &parser,
        .timeout_ms = 120000,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        return ESP_FAIL;
    }

    char auth_header[96];
    snprintf(auth_header, sizeof(auth_header), "Bearer %s", auth_token);
    esp_http_client_set_header(client, "Authorization", auth_header);
    esp_http_client_set_header(client, "Accept", "text/event-stream");

    esp_err_t err = esp_http_client_perform(client);
    esp_http_client_cleanup(client);
    ESP_LOGI(TAG, "SSE stream closed: %s", esp_err_to_name(err));
    return err;
}
