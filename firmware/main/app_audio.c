#include "app_audio.h"

#include <stdlib.h>
#include <string.h>

#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"

static const char *TAG = "audio";

#define MIC_BCLK GPIO_NUM_8
#define MIC_WS GPIO_NUM_7
#define MIC_DIN GPIO_NUM_9

#define SPK_BCLK GPIO_NUM_2
#define SPK_WS GPIO_NUM_3
#define SPK_DOUT GPIO_NUM_4

#define RECORD_BUFFER_BYTES (16000 * 2 * 12)
#define PLAYBACK_RING_BYTES (22050 * 2 * 8)
#define VAD_SILENCE_MS 900
#define VAD_FRAME_MS 30

static i2s_chan_handle_t s_mic_chan;
static i2s_chan_handle_t s_spk_chan;
static uint8_t *s_record_buf;
static size_t s_record_len;
static bool s_recording;
static RingbufHandle_t s_playback_ring;
static TaskHandle_t s_playback_task;
static int s_silence_frames;

static void playback_task(void *arg)
{
    uint8_t chunk[1024];
    size_t bytes_written = 0;
    while (true) {
        size_t item_size = 0;
        uint8_t *item = (uint8_t *)xRingbufferReceiveUpTo(s_playback_ring, &item_size, pdMS_TO_TICKS(50), sizeof(chunk));
        if (item && item_size > 0) {
            i2s_channel_write(s_spk_chan, item, item_size, &bytes_written, portMAX_DELAY);
            vRingbufferReturnItem(s_playback_ring, item);
        }
    }
}

static esp_err_t init_i2s_channels(void)
{
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &s_spk_chan, &s_mic_chan));

    i2s_std_config_t mic_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(LUMA7_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = MIC_BCLK,
            .ws = MIC_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = MIC_DIN,
        },
    };
    mic_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_mic_chan, &mic_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_mic_chan));

    i2s_std_config_t spk_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(LUMA7_PLAYBACK_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = SPK_BCLK,
            .ws = SPK_WS,
            .dout = SPK_DOUT,
            .din = I2S_GPIO_UNUSED,
        },
    };
    spk_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_spk_chan, &spk_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_spk_chan));
    return ESP_OK;
}

static size_t build_wav(uint8_t *pcm, size_t pcm_len, uint8_t **wav_out)
{
    const size_t header_size = 44;
    size_t total = header_size + pcm_len;
    uint8_t *wav = heap_caps_malloc(total, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!wav) {
        return 0;
    }
    memcpy(wav + header_size, pcm, pcm_len);
    const uint32_t data_size = (uint32_t)pcm_len;
    const uint32_t riff_size = 36 + data_size;
    const uint32_t sample_rate = LUMA7_SAMPLE_RATE;
    const uint16_t channels = 1;
    const uint16_t bits = 16;
    memcpy(wav + 0, "RIFF", 4);
    memcpy(wav + 4, &riff_size, 4);
    memcpy(wav + 8, "WAVE", 4);
    memcpy(wav + 12, "fmt ", 4);
    const uint32_t fmt_size = 16;
    memcpy(wav + 16, &fmt_size, 4);
    const uint16_t audio_format = 1;
    memcpy(wav + 20, &audio_format, 2);
    memcpy(wav + 22, &channels, 2);
    memcpy(wav + 24, &sample_rate, 4);
    const uint32_t byte_rate = sample_rate * channels * bits / 8;
    memcpy(wav + 28, &byte_rate, 4);
    const uint16_t block_align = channels * bits / 8;
    memcpy(wav + 32, &block_align, 2);
    memcpy(wav + 34, &bits, 2);
    memcpy(wav + 36, "data", 4);
    memcpy(wav + 40, &data_size, 4);
    *wav_out = wav;
    return total;
}

esp_err_t luma7_audio_init(void)
{
    s_record_buf = heap_caps_malloc(RECORD_BUFFER_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_record_buf) {
        return ESP_ERR_NO_MEM;
    }
    s_playback_ring = xRingbufferCreate(PLAYBACK_RING_BYTES, RINGBUF_TYPE_BYTEBUF);
    if (!s_playback_ring) {
        return ESP_ERR_NO_MEM;
    }
    ESP_ERROR_CHECK(init_i2s_channels());
    xTaskCreatePinnedToCore(playback_task, "playback", 4096, NULL, 6, &s_playback_task, 1);
    ESP_LOGI(TAG, "audio ready");
    return ESP_OK;
}

esp_err_t luma7_audio_start_record(void)
{
    s_record_len = 0;
    s_recording = true;
    s_silence_frames = 0;
    return ESP_OK;
}

static int16_t frame_energy(const int16_t *samples, size_t count)
{
    int64_t sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += abs(samples[i]);
    }
    return (int16_t)(sum / (int64_t)count);
}

void luma7_audio_poll_record(void)
{
    if (!s_recording || !s_record_buf) {
        return;
    }
    int16_t frame[480];
    size_t bytes_read = 0;
    if (i2s_channel_read(s_mic_chan, frame, sizeof(frame), &bytes_read, pdMS_TO_TICKS(10)) != ESP_OK) {
        return;
    }
    if (s_record_len + bytes_read > RECORD_BUFFER_BYTES) {
        s_recording = false;
        return;
    }
    memcpy(s_record_buf + s_record_len, frame, bytes_read);
    s_record_len += bytes_read;

    if (frame_energy(frame, bytes_read / sizeof(int16_t)) < 200) {
        s_silence_frames++;
    } else {
        s_silence_frames = 0;
    }
}

size_t luma7_audio_stop_record(uint8_t **wav_out)
{
    s_recording = false;
    if (!s_record_buf || s_record_len == 0 || !wav_out) {
        return 0;
    }
    return build_wav(s_record_buf, s_record_len, wav_out);
}

void luma7_audio_discard_record(void)
{
    s_recording = false;
    s_record_len = 0;
}

void luma7_audio_playback_push(const uint8_t *pcm, size_t len)
{
    if (!pcm || len == 0 || !s_playback_ring) {
        return;
    }
    xRingbufferSend(s_playback_ring, pcm, len, pdMS_TO_TICKS(20));
}

void luma7_audio_playback_flush(void)
{
    if (!s_playback_ring) {
        return;
    }
    size_t item_size = 0;
    uint8_t *item;
    while ((item = (uint8_t *)xRingbufferReceive(s_playback_ring, &item_size, 0)) != NULL) {
        vRingbufferReturnItem(s_playback_ring, item);
    }
}

void luma7_audio_playback_stop(void)
{
    luma7_audio_playback_flush();
}

bool luma7_audio_vad_silence_detected(void)
{
    const int frames = VAD_SILENCE_MS / VAD_FRAME_MS;
    return s_recording && s_silence_frames >= frames;
}
