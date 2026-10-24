/*
 * qwen_asr_ggml.h - encoder and decoder on a ggml device (Metal, Vulkan; --device)
 */

#ifndef QWEN_ASR_GGML_H
#define QWEN_ASR_GGML_H

#include "qwen_asr.h"

#ifdef USE_GGML
/* Print the ggml GPUs to stdout (--list-devices), marking the one
 * --device gpu picks; with json, as ",{...}" array elements. 0 on success. */
int qwen_ggml_list_devices(int json);

/* Upload the loaded decoder weights to the qwen_device GPU. 0 on success,
 * -1 when "gpu" finds none, -2 when no GPU has that name. */
int qwen_ggml_decoder_init(qwen_ctx_t *ctx);
void qwen_ggml_decoder_free(qwen_ctx_t *ctx);

/* Run n embeddings at positions kv_cache_len.. and append them to the device
 * KV cache. With want_logits, the logits of all n rows are kept for
 * qwen_ggml_logits(). Does not advance ctx->kv_cache_len. 0 on success. */
int qwen_ggml_decode(qwen_ctx_t *ctx, const float *embeds, int n, int want_logits);
const float *qwen_ggml_logits(qwen_ctx_t *ctx, int row);

/* Encoder on the same device; qwen_ggml_encoder_forward mirrors
 * qwen_encoder_forward (returns malloc'd [seq, output_dim] or NULL). */
int qwen_ggml_encoder_init(qwen_ctx_t *ctx);
float *qwen_ggml_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames, int *out_seq_len);
#endif

#endif /* QWEN_ASR_GGML_H */
