#include "app_http.h"

#include <stdio.h>
#include <string.h>

#include "cJSON.h"
#include "esp_http_client.h"
#include "esp_log.h"

static const char *TAG = "http";

typedef struct {
    char *buffer;
    size_t capacity;
    size_t length;
} response_accumulator_t;

static esp_err_t http_event_handler(esp_http_client_event_t *evt)
{
    response_accumulator_t *acc = (response_accumulator_t *)evt->user_data;
    if (evt->event_id == HTTP_EVENT_ON_DATA && acc && evt->data_len > 0) {
        if (acc->length + evt->data_len + 1 > acc->capacity) {
            size_t new_cap = acc->capacity + evt->data_len + 128;
            char *next = realloc(acc->buffer, new_cap);
            if (!next) {
                return ESP_FAIL;
            }
            acc->buffer = next;
            acc->capacity = new_cap;
        }
        memcpy(acc->buffer + acc->length, evt->data, evt->data_len);
        acc->length += evt->data_len;
        acc->buffer[acc->length] = '\0';
    }
    return ESP_OK;
}

static esp_err_t post_binary(const char *url, const char *auth_token, const uint8_t *body, size_t body_len, char **response_out)
{
    response_accumulator_t acc = {.buffer = calloc(1, 256), .capacity = 256, .length = 0};
    if (!acc.buffer) {
        return ESP_ERR_NO_MEM;
    }

    esp_http_client_config_t config = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .event_handler = http_event_handler,
        .user_data = &acc,
        .timeout_ms = 30000,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (!client) {
        free(acc.buffer);
        return ESP_FAIL;
    }

    char auth_header[96];
    snprintf(auth_header, sizeof(auth_header), "Bearer %s", auth_token);
    esp_http_client_set_header(client, "Authorization", auth_header);
    esp_http_client_set_header(client, "Content-Type", "application/octet-stream");
    esp_http_client_set_post_field(client, (const char *)body, body_len);

    esp_err_t err = esp_http_client_perform(client);
    int status = esp_http_client_get_status_code(client);
    esp_http_client_cleanup(client);

    if (err != ESP_OK || status < 200 || status >= 300) {
        ESP_LOGE(TAG, "POST failed status=%d err=%s body=%s", status, esp_err_to_name(err), acc.buffer ? acc.buffer : "");
        free(acc.buffer);
        return ESP_FAIL;
    }

    *response_out = acc.buffer;
    return ESP_OK;
}

esp_err_t luma7_http_post_query(
    const char *server_url,
    const char *auth_token,
    const uint8_t *jpeg,
    size_t jpeg_len,
    const uint8_t *wav,
    size_t wav_len,
    char *session_id_out,
    size_t session_id_len)
{
    if (!server_url || !auth_token || !wav || wav_len == 0 || !session_id_out) {
        return ESP_ERR_INVALID_ARG;
    }

    size_t payload_len = 4 + jpeg_len + wav_len;
    uint8_t *payload = malloc(payload_len);
    if (!payload) {
        return ESP_ERR_NO_MEM;
    }
    uint32_t be_len = __builtin_bswap32((uint32_t)jpeg_len);
    memcpy(payload, &be_len, 4);
    if (jpeg_len > 0 && jpeg) {
        memcpy(payload + 4, jpeg, jpeg_len);
    }
    memcpy(payload + 4 + jpeg_len, wav, wav_len);

    char url[192];
    snprintf(url, sizeof(url), "%s/query", server_url);

    char *response = NULL;
    esp_err_t err = post_binary(url, auth_token, payload, payload_len, &response);
    free(payload);
    if (err != ESP_OK) {
        return err;
    }

    cJSON *json = cJSON_Parse(response);
    free(response);
    if (!json) {
        return ESP_FAIL;
    }
    cJSON *session = cJSON_GetObjectItem(json, "session_id");
    if (!cJSON_IsString(session)) {
        cJSON_Delete(json);
        return ESP_FAIL;
    }
    strncpy(session_id_out, session->valuestring, session_id_len - 1);
    session_id_out[session_id_len - 1] = '\0';
    cJSON_Delete(json);
    ESP_LOGI(TAG, "session_id=%s", session_id_out);
    return ESP_OK;
}

esp_err_t luma7_http_post_stop(const char *server_url, const char *auth_token, const char *session_id)
{
    char url[192];
    snprintf(url, sizeof(url), "%s/stop/%s", server_url, session_id);
    char *response = NULL;
    esp_err_t err = post_binary(url, auth_token, (const uint8_t *)"", 0, &response);
    free(response);
    return err;
}
