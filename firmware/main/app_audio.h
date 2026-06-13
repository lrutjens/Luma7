#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#define LUMA7_SAMPLE_RATE 16000
#define LUMA7_PLAYBACK_RATE 22050

esp_err_t luma7_audio_init(void);
esp_err_t luma7_audio_start_record(void);
void luma7_audio_poll_record(void);
size_t luma7_audio_stop_record(uint8_t **wav_out);
void luma7_audio_discard_record(void);
void luma7_audio_playback_push(const uint8_t *pcm, size_t len);
void luma7_audio_playback_flush(void);
void luma7_audio_playback_stop(void);
bool luma7_audio_vad_silence_detected(void);
