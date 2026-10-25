/*
 * qwen_asr.c - Main API for Qwen3-ASR inference
 *
 * Pipeline: Load weights -> WAV -> Mel -> Encoder -> Build prompt ->
 *           Prefill decoder -> Autoregressive decode -> Tokenizer -> Text
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_ggml.h"
#include "qwen_asr_safetensors.h"
#include "qwen_asr_audio.h"
#include "qwen_asr_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>

/* Global verbose flag */
int qwen_verbose = 0;
const char *qwen_device = NULL;
int qwen_monitor = 0;

int qwen_list_devices(int json) {
    if (json)
        printf("[{\"name\":\"cpu\",\"description\":\"CPU\",\"type\":\"cpu\",\"threads\":%d,"
               "\"default\":true,\"gpu_default\":false}", qwen_get_threads());
    else
        printf("cpu: CPU (%d threads) [default]\n", qwen_get_threads());
#ifdef USE_GGML
    qwen_ggml_list_devices(json);
#endif
    if (json) printf("]\n");
    return 0;
}

void qwen_set_token_callback(qwen_ctx_t *ctx, qwen_token_cb cb, void *userdata) {
    ctx->token_cb = cb;
    ctx->token_cb_userdata = userdata;
}

void qwen_set_stream_update_callback(qwen_ctx_t *ctx, qwen_stream_update_cb cb, void *userdata) {
    ctx->stream_update_cb = cb;
    ctx->stream_update_cb_userdata = userdata;
}

static const char *QWEN_SUPPORTED_LANGUAGES[] = {
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch",
    "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"
};

static const char *QWEN_SUPPORTED_LANGUAGES_CSV =
    "Chinese,English,Cantonese,Arabic,German,French,Spanish,Portuguese,"
    "Indonesian,Italian,Korean,Russian,Thai,Vietnamese,Japanese,Turkish,"
    "Hindi,Malay,Dutch,Swedish,Danish,Finnish,Polish,Czech,Filipino,"
    "Persian,Greek,Romanian,Hungarian,Macedonian";

const char *qwen_supported_languages_csv(void) {
    return QWEN_SUPPORTED_LANGUAGES_CSV;
}

static void reset_prompt_cache(qwen_ctx_t *ctx) {
    free(ctx->prompt_tokens);
    ctx->prompt_tokens = NULL;
    ctx->n_prompt_tokens = 0;

    free(ctx->force_prompt_tokens);
    ctx->force_prompt_tokens = NULL;
    ctx->n_force_prompt_tokens = 0;

    ctx->prompt_tokens_ready = 0;
}

int qwen_set_prompt(qwen_ctx_t *ctx, const char *prompt) {
    if (!ctx) return -1;

    char *dup = NULL;
    if (prompt && prompt[0] != '\0') {
        dup = strdup(prompt);
        if (!dup) return -1;
    }
    free(ctx->prompt);
    ctx->prompt = dup;
    reset_prompt_cache(ctx);
    return 0;
}

static int normalize_language_name(const char *language, char *out, size_t out_cap) {
    if (!language || !out || out_cap < 2) return -1;

    while (*language && isspace((unsigned char)*language)) language++;
    size_t len = strlen(language);
    while (len > 0 && isspace((unsigned char)language[len - 1])) len--;
    if (len == 0 || len + 1 > out_cap) return -1;

    out[0] = (char)toupper((unsigned char)language[0]);
    for (size_t i = 1; i < len; i++) {
        out[i] = (char)tolower((unsigned char)language[i]);
    }
    out[len] = '\0';
    return 0;
}

static int is_supported_language(const char *language) {
    int n = (int)(sizeof(QWEN_SUPPORTED_LANGUAGES) / sizeof(QWEN_SUPPORTED_LANGUAGES[0]));
    for (int i = 0; i < n; i++) {
        if (strcmp(language, QWEN_SUPPORTED_LANGUAGES[i]) == 0) return 1;
    }
    return 0;
}

int qwen_set_force_language(qwen_ctx_t *ctx, const char *language) {
    if (!ctx) return -1;

    if (!language || language[0] == '\0') {
        free(ctx->force_language);
        ctx->force_language = NULL;
        reset_prompt_cache(ctx);
        return 0;
    }

    char normalized[64];
    if (normalize_language_name(language, normalized, sizeof(normalized)) != 0) return -1;
    if (!is_supported_language(normalized)) return -1;

    char *dup = strdup(normalized);
    if (!dup) return -1;

    free(ctx->force_language);
    ctx->force_language = dup;
    reset_prompt_cache(ctx);
    return 0;
}

/* ========================================================================
 * Internal load functions (defined in encoder/decoder .c files)
 * ======================================================================== */

extern int qwen_encoder_load(qwen_encoder_t *enc, multi_safetensors_t *ms,
                              const qwen_config_t *cfg);
extern int qwen_decoder_load(qwen_decoder_t *dec, multi_safetensors_t *ms,
                              const qwen_config_t *cfg);

/* ========================================================================
 * Config Detection
 * ======================================================================== */

/* Detect model variant from config.json or heuristics */
static int detect_config(qwen_ctx_t *ctx) {
    qwen_config_t *cfg = &ctx->config;

    /* Try to detect from number of shards:
     * 1.7B has 2 shards, 0.6B has 1 shard
     * But we can also check a specific weight shape. */

    multi_safetensors_t *ms = (multi_safetensors_t *)ctx->safetensors;

    /* -hf repos rename thinker.* to model.* */
    cfg->hf_layout = multi_safetensors_find(ms,
        "model.audio_tower.conv2d1.weight", NULL) != NULL;

    /* Encoder layer 18 exists only in 1.7B (24 layers vs 18) */
    const safetensor_t *test = multi_safetensors_find(ms, cfg->hf_layout ?
        "model.audio_tower.layers.18.self_attn.q_proj.weight" :
        "thinker.audio_tower.layers.18.self_attn.q_proj.weight", NULL);

    if (test) {
        /* 1.7B model */
        cfg->enc_d_model = 1024;
        cfg->enc_layers = 24;
        cfg->enc_heads = 16;
        cfg->enc_head_dim = 64;
        cfg->enc_ffn_dim = 4096;
        cfg->enc_output_dim = 2048;
        cfg->dec_hidden = 2048;
        cfg->dec_layers = 28;
        cfg->dec_heads = 16;
        cfg->dec_kv_heads = 8;
        cfg->dec_head_dim = 128;
        cfg->dec_intermediate = 6144;
        if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen3-ASR-1.7B\n");
    } else {
        /* 0.6B model */
        cfg->enc_d_model = 896;
        cfg->enc_layers = 18;
        cfg->enc_heads = 14;
        cfg->enc_head_dim = 64;
        cfg->enc_ffn_dim = 3584;
        cfg->enc_output_dim = 1024;
        cfg->dec_hidden = 1024;
        cfg->dec_layers = 28;
        cfg->dec_heads = 16;
        cfg->dec_kv_heads = 8;
        cfg->dec_head_dim = 128;
        cfg->dec_intermediate = 3072;
        if (qwen_verbose >= 1) fprintf(stderr, "Detected: Qwen3-ASR-0.6B\n");
    }

    /* Common parameters */
    cfg->enc_n_window = 50;
    cfg->enc_n_window_infer = 800;
    cfg->enc_chunk_size = cfg->enc_n_window * 2; /* 100 */
    cfg->enc_conv_proj_dim = QWEN_CONV_HIDDEN * 16; /* 7680 */
    cfg->vocab_size = QWEN_VOCAB_SIZE;
    cfg->dec_rms_norm_eps = 1e-6f;
    cfg->dec_rope_theta = 1e6f;

    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

qwen_ctx_t *qwen_load(const char *model_dir) {
    qwen_ctx_t *ctx = (qwen_ctx_t *)calloc(1, sizeof(qwen_ctx_t));
    if (!ctx) return NULL;
    snprintf(ctx->model_dir, sizeof(ctx->model_dir), "%s", model_dir);

    /* Open safetensors (multi-shard) */
    if (qwen_verbose >= 1)
        fprintf(stderr, "Loading model from %s\n", model_dir);

    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "qwen_load: cannot open safetensors in %s\n", model_dir);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = ms;

    /* Detect model configuration */
    detect_config(ctx);

    /* Load encoder weights */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading encoder weights...\n");
    if (qwen_encoder_load(&ctx->encoder, ms, &ctx->config) != 0) {
        fprintf(stderr, "qwen_load: failed to load encoder\n");
        qwen_free(ctx);
        return NULL;
    }

    /* Load decoder weights */
    if (qwen_verbose >= 1) fprintf(stderr, "Loading decoder weights...\n");
    if (qwen_decoder_load(&ctx->decoder, ms, &ctx->config) != 0) {
        fprintf(stderr, "qwen_load: failed to load decoder\n");
        qwen_free(ctx);
        return NULL;
    }
    int on_gpu = qwen_device && strcasecmp(qwen_device, "cpu") != 0;
#ifdef USE_GGML
    if (on_gpu) {
        int rc = qwen_ggml_decoder_init(ctx);
        if (rc == 0) {
            /* QWEN_GPU_ENC=0 keeps the encoder on the CPU (debug) */
            const char *enc_env = getenv("QWEN_GPU_ENC");
            ctx->ggml_enc = !(enc_env && enc_env[0] == '0') && qwen_ggml_encoder_init(ctx) == 0;
        } else if (rc == -2) {
            qwen_free(ctx);
            return NULL;
        } else {
            fprintf(stderr, "qwen_load: no usable GPU, running on the CPU\n");
        }
    }
#else
    if (on_gpu && strcasecmp(qwen_device, "gpu") == 0) {
        fprintf(stderr, "qwen_load: built without GPU support (make gpu), running on the CPU\n");
    } else if (on_gpu) {
        fprintf(stderr, "qwen_load: built without GPU support (make gpu), no device %s\n", qwen_device);
        qwen_free(ctx);
        return NULL;
    }
#endif

    /* Default transcription mode: full-audio offline decode (no splitting). */
    ctx->segment_sec = 0.0f;
    ctx->search_sec = 3.0f;

    /* Default streaming parameters */
    ctx->stream_chunk_sec = 2.0f;
    ctx->stream_rollback = 5;
    ctx->stream_unfixed_chunks = 2;
    ctx->stream_idle_flush_ms = 800;
    ctx->stream_pause_reset_ms = 3000;
    ctx->stream_max_new_tokens = 32;
    ctx->past_text_conditioning = 0;
    ctx->skip_silence = 0;

    if (qwen_verbose >= 1) fprintf(stderr, "Model loaded.\n");
    return ctx;
}

/* ========================================================================
 * Free
 * ======================================================================== */

void qwen_free(qwen_ctx_t *ctx) {
    if (!ctx) return;

    #define FREE0(p) do { free(p); (p) = NULL; } while (0)

    /* Encoder conv stem */
    FREE0(ctx->encoder.conv1_weight); FREE0(ctx->encoder.conv1_bias);
    FREE0(ctx->encoder.conv2_weight); FREE0(ctx->encoder.conv2_bias);
    FREE0(ctx->encoder.conv3_weight); FREE0(ctx->encoder.conv3_bias);
    FREE0(ctx->encoder.conv_out_weight);

    /* Encoder layers (weights are pre-converted f32, all allocated) */
    for (int i = 0; i < ctx->config.enc_layers; i++) {
        qwen_enc_layer_t *l = &ctx->encoder.layers[i];
        FREE0(l->wq_weight); FREE0(l->wq_bias);
        FREE0(l->wk_weight); FREE0(l->wk_bias);
        FREE0(l->wv_weight); FREE0(l->wv_bias);
        FREE0(l->wo_weight); FREE0(l->wo_bias);
        FREE0(l->attn_norm_weight); FREE0(l->attn_norm_bias);
        FREE0(l->fc1_weight); FREE0(l->fc1_bias);
        FREE0(l->fc2_weight); FREE0(l->fc2_bias);
        FREE0(l->ffn_norm_weight); FREE0(l->ffn_norm_bias);
    }
    FREE0(ctx->encoder.ln_post_weight); FREE0(ctx->encoder.ln_post_bias);
    FREE0(ctx->encoder.proj1_weight); FREE0(ctx->encoder.proj1_bias);
    FREE0(ctx->encoder.proj2_weight); FREE0(ctx->encoder.proj2_bias);

    /* Decoder layers */
    for (int i = 0; i < ctx->config.dec_layers; i++) {
        qwen_dec_layer_t *l = &ctx->decoder.layers[i];
        FREE0(l->q_norm_weight); FREE0(l->k_norm_weight);
        FREE0(l->input_norm); FREE0(l->post_attn_norm);
        FREE0(l->gate_up_fused_bf16);
    }
    FREE0(ctx->decoder.norm);

    #undef FREE0

#ifdef USE_GGML
    qwen_ggml_decoder_free(ctx);
#endif
    /* KV cache */
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);

    /* Persistent decoder buffers */
    free(ctx->dec_x); free(ctx->dec_x_norm);
    free(ctx->dec_q); free(ctx->dec_k); free(ctx->dec_v);
    free(ctx->dec_attn_out); free(ctx->dec_proj_out);
    free(ctx->dec_gate); free(ctx->dec_up); free(ctx->dec_ffn_out);
    free(ctx->dec_rope_cos); free(ctx->dec_rope_sin);

    /* Persistent decoder prefill buffers */
    free(ctx->pref_x); free(ctx->pref_x_norm);
    free(ctx->pref_q); free(ctx->pref_k); free(ctx->pref_v);
    free(ctx->pref_attn_out); free(ctx->pref_proj_out); free(ctx->pref_ffn_out);
    free(ctx->pref_gate); free(ctx->pref_gate_up);

    /* Decoder RoPE caches */
    free(ctx->rope_cache_cos); free(ctx->rope_cache_sin);
    free(ctx->rope_inv_freq);

    /* Prompt/language options */
    free(ctx->prompt);
    free(ctx->force_language);
    free(ctx->prompt_tokens);
    free(ctx->force_prompt_tokens);

    /* Close safetensors */
    if (ctx->safetensors) {
        multi_safetensors_close((multi_safetensors_t *)ctx->safetensors);
    }

    free(ctx);
}

/* ========================================================================
 * Transcription
 * ======================================================================== */

/*
 * Prompt format:
 *   PREFIX_HEAD: [<|im_start|>, "system", "\n"]
 *   [optional system prompt text tokens]
 *   PREFIX_TAIL: [<|im_end|>, "\n", <|im_start|>, "user", "\n", <|audio_start|>]
 *   AUDIO: [151676] × N_audio_tokens
 *   SUFFIX_BASE: [<|audio_end|>, <|im_end|>, "\n", <|im_start|>, "assistant", "\n"]
 *   [optional language tokens: "language X" + <asr_text>]
 */
static const int PROMPT_PREFIX_HEAD[] = {
    151644, 8948, 198
};
static const int PROMPT_PREFIX_TAIL[] = {
    151645, 198, 151644, 872, 198, 151669
};
static const int PROMPT_SUFFIX_BASE[] = {
    151670, 151645, 198, 151644, 77091, 198
};
#define PREFIX_HEAD_LEN 3
#define PREFIX_TAIL_LEN 6
#define SUFFIX_BASE_LEN 6

/* Convert a single token embedding from bf16 to f32 */
static void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

/* Drop long silent spans while preserving short pauses for readability.
 * Uses adaptive RMS gating with spike rejection for noisy backgrounds. */
static float *compact_silence(const float *samples, int n_samples, int *out_samples) {
    if (!samples || n_samples <= 0 || !out_samples) return NULL;

    const int win = 160;               /* 10 ms at 16kHz */
    const float base_thresh = 0.002f;  /* ~ -54 dBFS */
    const float max_thresh = 0.025f;   /* avoid over-aggressive clipping */
    const float smooth_alpha = 0.2f;   /* smooth frame-level RMS */
    const int min_voice_windows = 5;   /* reject <50ms spikes as noise */
    const int pad_voice_windows = 3;   /* keep 30ms around speech edges */
    const int pass_windows = 60;       /* keep first 600ms of silence */

    int n_win = (n_samples + win - 1) / win;
    float *rms_vals = (float *)malloc((size_t)n_win * sizeof(float));
    float *sorted = (float *)malloc((size_t)n_win * sizeof(float));
    float *smooth_vals = (float *)malloc((size_t)n_win * sizeof(float));
    unsigned char *is_voice = (unsigned char *)malloc((size_t)n_win);
    if (!rms_vals || !sorted || !smooth_vals || !is_voice) {
        free(rms_vals);
        free(sorted);
        free(smooth_vals);
        free(is_voice);
        return NULL;
    }

    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;
        float energy = 0.0f;
        for (int i = 0; i < len; i++) {
            float v = samples[start + i];
            energy += v * v;
        }
        rms_vals[w] = sqrtf(energy / (float)(len > 0 ? len : 1));
    }

    /* Smooth RMS so tiny impulsive noise does not flip decisions. */
    float smooth = rms_vals[0];
    for (int w = 0; w < n_win; w++) {
        smooth = (1.0f - smooth_alpha) * smooth + smooth_alpha * rms_vals[w];
        smooth_vals[w] = smooth;
    }

    memcpy(sorted, smooth_vals, (size_t)n_win * sizeof(float));
    qsort(sorted, (size_t)n_win, sizeof(float), cmp_float_asc);

    /* Adaptive threshold from low-energy percentile (robust to loud clips). */
    int p25 = (int)((n_win - 1) * 0.25f);
    float noise_floor = sorted[p25];
    float thresh = noise_floor * 1.8f;
    if (thresh < base_thresh) thresh = base_thresh;
    if (thresh > max_thresh) thresh = max_thresh;
    free(sorted);

    for (int w = 0; w < n_win; w++) {
        is_voice[w] = (smooth_vals[w] > thresh) ? 1 : 0;
    }
    free(smooth_vals);

    /* Remove very short voice bursts (usually clicks/hiss spikes). */
    for (int i = 0; i < n_win; ) {
        if (!is_voice[i]) { i++; continue; }
        int j = i + 1;
        while (j < n_win && is_voice[j]) j++;
        if (j - i < min_voice_windows) {
            memset(is_voice + i, 0, (size_t)(j - i));
        }
        i = j;
    }

    /* Add a small speech edge pad to avoid clipping word boundaries. */
    unsigned char *padded = (unsigned char *)calloc((size_t)n_win, 1);
    if (!padded) {
        free(is_voice);
        free(rms_vals);
        return NULL;
    }
    for (int w = 0; w < n_win; w++) {
        if (!is_voice[w]) continue;
        int a = w - pad_voice_windows;
        int b = w + pad_voice_windows;
        if (a < 0) a = 0;
        if (b >= n_win) b = n_win - 1;
        for (int k = a; k <= b; k++) padded[k] = 1;
    }
    free(is_voice);

    float *out = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!out) {
        free(rms_vals);
        free(padded);
        return NULL;
    }

    int out_n = 0;
    int silence_count = 0;
    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;

        if (padded[w]) {
            memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
            out_n += len;
            silence_count = 0;
        } else {
            silence_count++;
            if (silence_count <= pass_windows) {
                memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
                out_n += len;
            }
        }
    }
    free(padded);
    free(rms_vals);

    if (out_n == 0) {
        int keep = n_samples;
        int min_keep = QWEN_SAMPLE_RATE / 2;
        if (keep > min_keep) keep = min_keep;
        memcpy(out, samples, (size_t)keep * sizeof(float));
        out_n = keep;
    }

    *out_samples = out_n;
    return out;
}

/* Prepare cached prompt-related tokens once per context. */
static int prepare_prompt_tokens(qwen_ctx_t *ctx, qwen_tokenizer_t *tokenizer) {
    if (ctx->prompt_tokens_ready) return 0;

    reset_prompt_cache(ctx);

    if (ctx->prompt && ctx->prompt[0] != '\0') {
        ctx->prompt_tokens = qwen_tokenizer_encode(tokenizer, ctx->prompt, &ctx->n_prompt_tokens);
        if (!ctx->prompt_tokens) {
            fprintf(stderr, "qwen: failed to encode --prompt text\n");
            return -1;
        }
    }

    if (ctx->force_language && ctx->force_language[0] != '\0') {
        char force_text[128];
        snprintf(force_text, sizeof(force_text), "language %s", ctx->force_language);

        int n_lang_txt = 0;
        int *lang_txt_tokens = qwen_tokenizer_encode(tokenizer, force_text, &n_lang_txt);
        if (!lang_txt_tokens) {
            fprintf(stderr, "qwen: failed to encode --language text\n");
            return -1;
        }

        ctx->n_force_prompt_tokens = n_lang_txt + 1; /* + <asr_text> marker */
        ctx->force_prompt_tokens = (int *)malloc((size_t)ctx->n_force_prompt_tokens * sizeof(int));
        if (!ctx->force_prompt_tokens) {
            free(lang_txt_tokens);
            return -1;
        }
        if (n_lang_txt > 0) {
            memcpy(ctx->force_prompt_tokens, lang_txt_tokens, (size_t)n_lang_txt * sizeof(int));
        }
        ctx->force_prompt_tokens[n_lang_txt] = QWEN_TOKEN_ASR_TEXT;
        free(lang_txt_tokens);
    }

    ctx->prompt_tokens_ready = 1;
    return 0;
}

/* ---- Segment-based transcription ---- */

#define ENERGY_WINDOW_MS    100

/*
 * Find the best split point near target_sample by looking for the
 * lowest-energy 100ms window within +/-search_sec seconds.
 */
static int find_split_point(const float *samples, int n_samples,
                            int target_sample, float search_sec) {
    int search_half = (int)(search_sec * QWEN_SAMPLE_RATE);
    int lo = target_sample - search_half;
    int hi = target_sample + search_half;
    if (lo < 0) lo = 0;
    if (hi > n_samples) hi = n_samples;

    int win_samples = (ENERGY_WINDOW_MS * QWEN_SAMPLE_RATE) / 1000; /* 1600 */
    float best_energy = 1e30f;
    int best_center = target_sample;

    for (int pos = lo; pos + win_samples <= hi; pos += win_samples / 2) {
        float energy = 0;
        int end = pos + win_samples;
        if (end > n_samples) end = n_samples;
        for (int j = pos; j < end; j++) {
            energy += samples[j] * samples[j];
        }
        energy /= (end - pos);
        if (energy < best_energy) {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
    }
    return best_center;
}

/*
 * Transcribe a single audio segment. Returns malloc'd text or NULL.
 * The tokenizer is passed in so we only load it once.
 */
static char *transcribe_segment(qwen_ctx_t *ctx, const float *samples,
                                int n_samples, qwen_tokenizer_t *tokenizer,
                                const int *past_tokens, int n_past_tokens,
                                int *out_text_tokens) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    double seg_t0 = get_time_ms();
    int n_text_tokens = 0;

    /* ---- Mel spectrogram ---- */
    double t0 = get_time_ms();
    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames);
    if (!mel) return NULL;
    double mel_ms = get_time_ms() - t0;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Mel: %d frames (%.0f ms)\n", mel_frames, mel_ms);

    /* ---- Encoder ---- */
    t0 = get_time_ms();
    int enc_seq_len = 0;
    float *enc_output = qwen_encoder_forward(ctx, mel, mel_frames, &enc_seq_len);
    free(mel);
    if (!enc_output) return NULL;
    double enc_ms = get_time_ms() - t0;

    if (qwen_verbose >= 2)
        fprintf(stderr, "  Encoder: %d tokens (%.0f ms)\n", enc_seq_len, enc_ms);

    if (prepare_prompt_tokens(ctx, tokenizer) != 0) {
        free(enc_output);
        return NULL;
    }

    /* ---- Build input embeddings ---- */
    int prefix_len = PREFIX_HEAD_LEN + ctx->n_prompt_tokens + PREFIX_TAIL_LEN;
    int suffix_len = SUFFIX_BASE_LEN + ctx->n_force_prompt_tokens;
    int n_past_prompt_tokens = (n_past_tokens > 0) ? (n_past_tokens + 1) : 0; /* + <asr_text> */
    int total_seq = prefix_len + enc_seq_len + suffix_len + n_past_prompt_tokens;
    float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!input_embeds || !tmp_embed) {
        free(enc_output);
        free(input_embeds);
        free(tmp_embed);
        return NULL;
    }

    /* Embed prefix head: <|im_start|>system\n */
    int off = 0;
    for (int i = 0; i < PREFIX_HEAD_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              PROMPT_PREFIX_HEAD[i], dim);
        off++;
    }

    /* Embed optional prompt text (system content) */
    for (int i = 0; i < ctx->n_prompt_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              ctx->prompt_tokens[i], dim);
        off++;
    }

    /* Embed prefix tail: <|im_end|>\n<|im_start|>user\n<|audio_start|> */
    for (int i = 0; i < PREFIX_TAIL_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + off * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              PROMPT_PREFIX_TAIL[i], dim);
        off++;
    }

    /* Replace audio_pad positions with encoder output */
    for (int i = 0; i < enc_seq_len; i++) {
        memcpy(input_embeds + (prefix_len + i) * dim,
               enc_output + i * dim,
               dim * sizeof(float));
    }
    free(enc_output);

    /* Embed suffix base: <|audio_end|><|im_end|>\n<|im_start|>assistant\n */
    int suffix_off = prefix_len + enc_seq_len;
    for (int i = 0; i < SUFFIX_BASE_LEN; i++) {
        tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              PROMPT_SUFFIX_BASE[i], dim);
    }

    /* Optional forced-language suffix: "language X" + <asr_text> */
    for (int i = 0; i < ctx->n_force_prompt_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + (suffix_off + SUFFIX_BASE_LEN + i) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              ctx->force_prompt_tokens[i], dim);
    }

    /* Optional past-text conditioning tokens (for segmented mode).
     * Put a fresh <asr_text> marker AFTER the past text so generation
     * restarts from a new ASR span instead of terminating immediately. */
    int past_off = suffix_off + suffix_len;
    for (int i = 0; i < n_past_tokens; i++) {
        tok_embed_bf16_to_f32(input_embeds + (past_off + i) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              past_tokens[i], dim);
    }
    if (n_past_tokens > 0) {
        tok_embed_bf16_to_f32(input_embeds + (past_off + n_past_tokens) * dim,
                              ctx->decoder.tok_embeddings_bf16,
                              QWEN_TOKEN_ASR_TEXT, dim);
    }

    /* ---- Decoder prefill ---- */
    t0 = get_time_ms();
    ctx->kv_cache_len = 0; /* Reset KV cache for this segment */
    int prefill_len = total_seq - 1; /* prefill all but last */
    qwen_decoder_prefill(ctx, input_embeds, prefill_len);

    /* First token from last prefill position */
    float *last_embed = input_embeds + (size_t)prefill_len * dim;
    int token = qwen_decoder_forward(ctx, last_embed);
    free(input_embeds);

    double prefill_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Prefill: %d tokens (%.0f ms)\n", total_seq, prefill_ms);

    /* ---- Autoregressive decode ---- */
    t0 = get_time_ms();
    int max_tokens = 2048;
    int n_generated = 0;
    /* If language is forced, <asr_text> is already part of prompt suffix. */
    int past_asr_text = (ctx->n_force_prompt_tokens > 0 || n_past_tokens > 0) ? 1 : 0;

    size_t text_cap = 4096;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    while (n_generated < max_tokens) {
        n_generated++;

        /* Check EOS */
        if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;

        /* Track <asr_text> marker */
        if (token == QWEN_TOKEN_ASR_TEXT) {
            past_asr_text = 1;
        } else if (past_asr_text) {
            /* Decode and emit this text token */
            const char *piece = qwen_tokenizer_decode(tokenizer, token);
            size_t piece_len = strlen(piece);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)realloc(text, text_cap);
            }
            memcpy(text + text_len, piece, piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
            n_text_tokens++;

            /* Stream token via callback */
            if (ctx->token_cb)
                ctx->token_cb(piece, ctx->token_cb_userdata);
        }

        /* Embed and generate next token */
        tok_embed_bf16_to_f32(tmp_embed, ctx->decoder.tok_embeddings_bf16, token, dim);
        token = qwen_decoder_forward(ctx, tmp_embed);
    }

    double decode_ms = get_time_ms() - t0;
    if (qwen_verbose >= 2)
        fprintf(stderr, "  Decode: %d tokens (%.0f ms, %.1f ms/token)\n",
                n_generated, decode_ms,
                n_generated > 0 ? decode_ms / n_generated : 0);

    free(tmp_embed);

    /* Trim whitespace */
    size_t rlen = strlen(text);
    while (rlen > 0 && isspace((unsigned char)text[rlen - 1])) text[--rlen] = '\0';
    char *start = text;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != text) memmove(text, start, strlen(start) + 1);

    ctx->perf_total_ms += get_time_ms() - seg_t0;
    ctx->perf_text_tokens += n_text_tokens;
    ctx->perf_encode_ms += mel_ms + enc_ms;
    ctx->perf_decode_ms += prefill_ms + decode_ms;
    if (out_text_tokens) *out_text_tokens = n_text_tokens;

    return text;
}

static int should_retry_unconditioned_segment(const char *full_result,
                                              const char *seg_text,
                                              int core_samples,
                                              int n_text_tokens) {
    if (!seg_text || seg_text[0] == '\0') return 1;

    /* A segment producing very few tokens under conditioning is usually
     * a collapse (model repeats/terminates early instead of following audio).
     * Use stricter checks from ~8s upward to catch common -S 10 failures. */
    float core_sec = (float)core_samples / (float)QWEN_SAMPLE_RATE;
    if (core_sec >= 8.0f) {
        int min_tokens = (int)(core_sec * 1.75f);
        if (min_tokens < 12) min_tokens = 12;
        if (n_text_tokens < min_tokens) return 1;
    }

    /* Exact duplicate span already present in accumulated text: likely drift. */
    if (full_result && full_result[0] != '\0') {
        size_t seg_len = strlen(seg_text);
        if (seg_len >= 48 && strstr(full_result, seg_text) != NULL) return 1;
    }

    return 0;
}

static int should_insert_boundary_space(int prev_ch, int next_ch) {
    if (prev_ch <= 0 || next_ch <= 0) return 0;
    if (isspace((unsigned char)prev_ch)) return 0;
    if (isspace((unsigned char)next_ch)) return 0;
    if (ispunct((unsigned char)next_ch)) return 0;
    return 1;
}

/* After punctuation, a word of a script written with spaces between words
 * (Latin, Cyrillic, Hangul, ...) starts with a space. Not CJK, kana, Thai,
 * Lao, Myanmar or Khmer, which are written without them. */
static int stream_seam_needs_space(int prev_ch, const char *next) {
    if (prev_ch <= 0 || !strchr(".,?!;:", prev_ch) || !next || !*next) return 0;
    const unsigned char *u = (const unsigned char *)next;
    unsigned cp;
    if (u[0] < 0x80) return isalpha(u[0]) != 0;
    if ((u[0] & 0xE0) == 0xC0 && u[1]) cp = ((u[0] & 0x1Fu) << 6) | (u[1] & 0x3Fu);
    else if ((u[0] & 0xF0) == 0xE0 && u[1] && u[2])
        cp = ((u[0] & 0x0Fu) << 12) | ((u[1] & 0x3Fu) << 6) | (u[2] & 0x3Fu);
    else return 0;
    if (cp < 0xC0) return 0;                                  /* Latin-1 symbols */
    if (cp >= 0x0E00 && cp <= 0x0EFF) return 0;               /* Thai, Lao */
    if (cp >= 0x1000 && cp <= 0x109F) return 0;               /* Myanmar */
    if (cp >= 0x1780 && cp <= 0x17FF) return 0;               /* Khmer */
    if (cp >= 0x2E80 && cp <= 0x9FFF) return 0;               /* CJK, kana */
    if (cp >= 0xF900 && cp <= 0xFAFF) return 0;
    if (cp >= 0xFF00 && cp <= 0xFFEF) return 0;               /* full-width forms */
    return 1;
}

static int stream_piece_ends_sentence(const char *piece) {
    size_t n = piece ? strlen(piece) : 0;
    if (n == 0) return 0;
    char c = piece[n - 1];
    if (c == '.' || c == '!' || c == '?') return 1;
    if (n >= 3) {
        const char *t = piece + n - 3;
        if (!memcmp(t, "\xe3\x80\x82", 3) ||       /* 。 */
            !memcmp(t, "\xef\xbc\x81", 3) ||       /* ！ */
            !memcmp(t, "\xef\xbc\x9f", 3) ||       /* ？ */
            !memcmp(t, "\xef\xbc\x8e", 3)) return 1; /* ． */
    }
    return 0;
}

static int stream_sec_to_chunks(double sec, double chunk_sec) {
    int n = (int)(sec / chunk_sec + 0.5);
    return n < 1 ? 1 : n;
}

typedef struct {
    qwen_token_cb downstream_cb;
    void *downstream_userdata;
    int maybe_prepend_space;
    int saw_first_piece;
} segment_emit_state_t;

static void segment_emit_cb(const char *piece, void *userdata) {
    segment_emit_state_t *st = (segment_emit_state_t *)userdata;
    if (!st || !st->downstream_cb || !piece) return;

    if (!st->saw_first_piece) {
        st->saw_first_piece = 1;
        if (st->maybe_prepend_space) {
            unsigned char c0 = (unsigned char)piece[0];
            if (c0 != '\0' && !isspace(c0) && !ispunct(c0)) {
                st->downstream_cb(" ", st->downstream_userdata);
            }
        }
    }
    st->downstream_cb(piece, st->downstream_userdata);
}

char *qwen_transcribe_audio(qwen_ctx_t *ctx, const float *samples, int n_samples) {
    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = 1000.0 * (double)n_samples / (double)QWEN_SAMPLE_RATE;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;

    const float *audio_samples = samples;
    int audio_n_samples = n_samples;
    float *compacted_samples = NULL;
    if (ctx->skip_silence) {
        compacted_samples = compact_silence(samples, n_samples, &audio_n_samples);
        if (compacted_samples) audio_samples = compacted_samples;
        if (qwen_verbose >= 1) {
            float used_pct = 100.0f * (float)audio_n_samples /
                             (float)(n_samples > 0 ? n_samples : 1);
            float skipped_pct = 100.0f - used_pct;
            if (skipped_pct < 0.0f) skipped_pct = 0.0f;
            fprintf(stderr, "Silence skip: used %.1f%%, skipped %.1f%% (%d -> %d samples)\n",
                    used_pct, skipped_pct, n_samples, audio_n_samples);
        }
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                audio_n_samples, (float)audio_n_samples / QWEN_SAMPLE_RATE);

    /* Load tokenizer once for all segments */
    qwen_tokenizer_t *tokenizer = qwen_tokenizer_load(ctx->model_dir);
    if (!tokenizer) {
        free(compacted_samples);
        return NULL;
    }
    if (prepare_prompt_tokens(ctx, tokenizer) != 0) {
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }

    /* Determine segment boundaries.
     * Clamp search window to half the segment size so split points
     * can never overlap and produce zero-length segments. */
    float search = ctx->search_sec;
    if (search > ctx->segment_sec / 2.0f) search = ctx->segment_sec / 2.0f;
    int target_samples = (int)(ctx->segment_sec * QWEN_SAMPLE_RATE);
    int margin_samples = (int)(search * QWEN_SAMPLE_RATE);

    /* No splitting if segment_sec is 0 or audio fits in one segment */
    if (ctx->segment_sec <= 0 || audio_n_samples <= target_samples + margin_samples) {
        char *text = transcribe_segment(ctx, audio_samples, audio_n_samples, tokenizer, NULL, 0, NULL);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return text;
    }

    /* Build split points */
    int splits[128]; /* max 128 segments */
    int n_splits = 0;
    splits[n_splits++] = 0;

    int pos = 0;
    while (pos + target_samples + margin_samples < audio_n_samples) {
        int split = find_split_point(audio_samples, audio_n_samples,
                                     pos + target_samples, search);
        splits[n_splits++] = split;
        pos = split;
        if (n_splits >= 127) break; /* safety */
    }
    splits[n_splits] = audio_n_samples; /* end sentinel */

    if (qwen_verbose >= 2)
        fprintf(stderr, "Splitting into %d segments\n", n_splits);

    /* Transcribe each segment and concatenate */
    size_t result_cap = 4096;
    size_t result_len = 0;
    char *result = (char *)malloc(result_cap);
    result[0] = '\0';
    int min_samples = QWEN_SAMPLE_RATE / 2; /* 0.5s minimum, like official */
    int do_boundary_cleanup = (ctx->past_text_conditioning != 0);
    int use_past_conditioning = ctx->past_text_conditioning;
    int conditioning_collapses = 0;
    qwen_token_cb saved_cb = ctx->token_cb;
    void *saved_cb_userdata = ctx->token_cb_userdata;

    for (int s = 0; s < n_splits; s++) {
        int core_start = splits[s];
        int core_end = splits[s + 1];
        int seg_start = core_start;
        int seg_end = core_end;
        int seg_samples = seg_end - seg_start;

        if (qwen_verbose >= 2)
            fprintf(stderr, "Segment %d/%d: core %.1f-%.1fs, decode %.1f-%.1fs (%d samples)\n",
                    s + 1, n_splits,
                    (float)core_start / QWEN_SAMPLE_RATE,
                    (float)core_end / QWEN_SAMPLE_RATE,
                    (float)seg_start / QWEN_SAMPLE_RATE,
                    (float)seg_end / QWEN_SAMPLE_RATE,
                    seg_samples);

        /* Pad short segments to 0.5s with zeros (like official pipeline) */
        float *seg_buf = NULL;
        const float *seg_ptr = audio_samples + seg_start;
        if (seg_samples < min_samples) {
            seg_buf = (float *)calloc(min_samples, sizeof(float));
            memcpy(seg_buf, seg_ptr, seg_samples * sizeof(float));
            seg_ptr = seg_buf;
            seg_samples = min_samples;
        }

        int *past_tokens = NULL;
        int n_past_tokens = 0;
        if (use_past_conditioning && result_len > 0) {
            past_tokens = qwen_tokenizer_encode(tokenizer, result, &n_past_tokens);
            if (!past_tokens) n_past_tokens = 0;
        }

        segment_emit_state_t emit_state = {0};
        if (do_boundary_cleanup) {
            /* Cleanup mode buffers segment output and emits finalized text only. */
            ctx->token_cb = NULL;
            ctx->token_cb_userdata = NULL;
        } else if (saved_cb) {
            /* Fast segmented mode: emit each generated token immediately.
             * Add one separating space before the first token of the segment
             * only when needed and only if the first piece does not already
             * begin with whitespace/punctuation. */
            emit_state.downstream_cb = saved_cb;
            emit_state.downstream_userdata = saved_cb_userdata;
            emit_state.maybe_prepend_space =
                (result_len > 0 && !isspace((unsigned char)result[result_len - 1]));
            emit_state.saw_first_piece = 0;
            ctx->token_cb = segment_emit_cb;
            ctx->token_cb_userdata = &emit_state;
        }

        int seg_text_tokens = 0;
        char *seg_text = transcribe_segment(ctx, seg_ptr, seg_samples, tokenizer,
                                            past_tokens, n_past_tokens,
                                            &seg_text_tokens);
        if (do_boundary_cleanup &&
            use_past_conditioning && n_past_tokens > 0 &&
            should_retry_unconditioned_segment(result, seg_text,
                                               core_end - core_start,
                                               seg_text_tokens)) {
            conditioning_collapses++;
            if (qwen_verbose >= 2) {
                fprintf(stderr,
                        "Segment mode: retrying segment %d/%d without past-text conditioning "
                        "(core=%.1fs, tokens=%d)\n",
                        s + 1, n_splits,
                        (float)(core_end - core_start) / QWEN_SAMPLE_RATE,
                        seg_text_tokens);
            }
            /* Guardrail: if conditioned decode collapses or drifts,
             * retry this segment without past-text conditioning. */
            free(seg_text);
            seg_text = transcribe_segment(ctx, seg_ptr, seg_samples, tokenizer, NULL, 0,
                                          &seg_text_tokens);
            if (conditioning_collapses >= 2) {
                use_past_conditioning = 0;
                if (qwen_verbose >= 2) {
                    fprintf(stderr, "Segment mode: disabling past text conditioning after %d collapses\n",
                            conditioning_collapses);
                }
            }
        }
        ctx->token_cb = saved_cb;
        ctx->token_cb_userdata = saved_cb_userdata;

        free(past_tokens);
        free(seg_buf);
        if (!seg_text) continue;
        if (seg_text[0] == '\0') { free(seg_text); continue; }

        int cut_pos = 0;
        if (do_boundary_cleanup) {
            while (seg_text[cut_pos] != '\0' && isspace((unsigned char)seg_text[cut_pos])) cut_pos++;
        }
        if (seg_text[cut_pos] == '\0') {
            free(seg_text);
            continue;
        }

        size_t add_len = strlen(seg_text + cut_pos);
        int need_space = should_insert_boundary_space(
            result_len > 0 ? (int)(unsigned char)result[result_len - 1] : 0,
            (int)(unsigned char)seg_text[cut_pos]);
        size_t need = result_len + add_len + (size_t)(need_space ? 2 : 1);
        if (need > result_cap) {
            size_t cap = result_cap;
            while (need > cap) cap *= 2;
            char *grown = (char *)realloc(result, cap);
            if (!grown) { free(seg_text); continue; }
            result = grown;
            result_cap = cap;
        }
        if (need_space) {
            result[result_len++] = ' ';
            if (do_boundary_cleanup && saved_cb) saved_cb(" ", saved_cb_userdata);
        }
        memcpy(result + result_len, seg_text + cut_pos, add_len);
        result_len += add_len;
        result[result_len] = '\0';
        if (do_boundary_cleanup && saved_cb) saved_cb(seg_text + cut_pos, saved_cb_userdata);
        free(seg_text);
    }

    ctx->token_cb = saved_cb;
    ctx->token_cb_userdata = saved_cb_userdata;
    qwen_tokenizer_free(tokenizer);
    free(compacted_samples);
    return result;
}

/* Encode one audio span into encoder tokens. Caller owns out_enc_output. */
static int stream_encode_span(qwen_ctx_t *ctx, const float *samples, int n_samples,
                              float **out_enc_output, int *out_seq_len) {
    *out_enc_output = NULL;
    *out_seq_len = 0;
    if (n_samples <= 0) return 0;

    int mel_frames = 0;
    float *mel = qwen_mel_spectrogram(samples, n_samples, &mel_frames);
    if (!mel) return -1;

    int seq_len = 0;
    float *enc_output = qwen_encoder_forward(ctx, mel, mel_frames, &seq_len);
    free(mel);
    if (!enc_output) return -1;

    *out_enc_output = enc_output;
    *out_seq_len = seq_len;
    return 0;
}

/* Detect repeated token blocks at the sequence tail.
 * Returns max repetitions found (>=1), and stores period in out_period. */
static int stream_tail_repeat_blocks(const int *tokens, int n_tokens, int max_period,
                                     int *out_period) {
    if (out_period) *out_period = 0;
    if (!tokens || n_tokens < 2) return 1;

    int best_reps = 1;
    int best_period = 0;
    int period_cap = n_tokens / 2;
    if (max_period > 0 && period_cap > max_period) period_cap = max_period;

    for (int p = 1; p <= period_cap; p++) {
        int reps = 1;
        while ((reps + 1) * p <= n_tokens) {
            const int *a = tokens + n_tokens - (reps + 1) * p;
            const int *b = tokens + n_tokens - reps * p;
            if (memcmp(a, b, (size_t)p * sizeof(int)) != 0) break;
            reps++;
        }
        if (reps > best_reps) {
            best_reps = reps;
            best_period = p;
        }
    }

    if (out_period) *out_period = best_period;
    return best_reps;
}

typedef struct {
    int64_t start_sample;
    int n_samples;
    int seq_len;
    float *enc_output; /* [seq_len, dec_hidden] */
} stream_enc_window_t;

static void stream_clear_enc_cache(stream_enc_window_t *enc_cache,
                                   int *n_enc_cache,
                                   int *enc_cache_start,
                                   int *enc_cached_seq_total,
                                   int64_t *next_window_start,
                                   int64_t new_start_sample) {
    if (!enc_cache || !n_enc_cache || !enc_cache_start ||
        !enc_cached_seq_total || !next_window_start) {
        return;
    }
    for (int i = *enc_cache_start; i < *n_enc_cache; i++) {
        free(enc_cache[i].enc_output);
        enc_cache[i].enc_output = NULL;
    }
    *n_enc_cache = 0;
    *enc_cache_start = 0;
    *enc_cached_seq_total = 0;
    *next_window_start = new_start_sample;
}

/* Length of the longest prefix of s[0..n) that does not end inside a UTF-8
 * sequence. Tokens are byte-level, so a character can span two tokens. */
static size_t stream_utf8_complete_len(const char *s, size_t n) {
    size_t back = 0;
    while (back < n && back < 4) {
        unsigned char c = (unsigned char)s[n - 1 - back];
        if ((c & 0xC0) != 0x80) {
            size_t need = 1;
            if ((c & 0xE0) == 0xC0) need = 2;
            else if ((c & 0xF0) == 0xE0) need = 3;
            else if ((c & 0xF8) == 0xF0) need = 4;
            return (back + 1 >= need) ? n : n - 1 - back;
        }
        back++;
    }
    return n;
}

/* Growable byte buffer for stream update strings. */
typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} stream_str_t;

static void stream_str_append(stream_str_t *s, const char *data, size_t n) {
    if (s->len + n + 1 > s->cap) {
        size_t new_cap = s->cap > 0 ? s->cap : 256;
        while (s->len + n + 1 > new_cap) new_cap *= 2;
        char *tmp = (char *)realloc(s->buf, new_cap);
        if (!tmp) return;
        s->buf = tmp;
        s->cap = new_cap;
    }
    memcpy(s->buf + s->len, data, n);
    s->len += n;
    s->buf[s->len] = '\0';
}

/* Drop the oldest cached encoder windows until at most `keep` remain. */
static int stream_trim_enc_cache(stream_enc_window_t *enc_cache,
                                 int n_enc_cache,
                                 int *enc_cache_start,
                                 int *enc_cached_seq_total,
                                 int keep) {
    int evicted = 0;
    while (n_enc_cache - *enc_cache_start > keep) {
        *enc_cached_seq_total -= enc_cache[*enc_cache_start].seq_len;
        free(enc_cache[*enc_cache_start].enc_output);
        enc_cache[*enc_cache_start].enc_output = NULL;
        (*enc_cache_start)++;
        evicted++;
    }
    return evicted;
}

/* Re-anchor stream text state to a short committed tail so decoding can
 * continue after a hard reset without replaying the full text history. */
static int stream_reanchor_text_state(qwen_ctx_t *ctx,
                                      const int *emitted_text_tokens,
                                      int n_emitted_text_tokens,
                                      int carry_text_tokens,
                                      int **raw_tokens,
                                      int *raw_tokens_cap,
                                      int *n_raw_tokens,
                                      int **stable_text_tokens,
                                      int *stable_text_cap,
                                      int *n_stable_text_tokens) {
    if (!ctx || !raw_tokens || !raw_tokens_cap || !n_raw_tokens ||
        !stable_text_tokens || !stable_text_cap || !n_stable_text_tokens) {
        return -1;
    }

    int carry = n_emitted_text_tokens;
    if (carry_text_tokens > 0 && carry > carry_text_tokens) carry = carry_text_tokens;
    if (carry < 0) carry = 0;

    int raw_lead = (ctx->n_force_prompt_tokens <= 0) ? 1 : 0; /* add <asr_text> marker */
    int raw_need = raw_lead + carry;

    if (raw_need > *raw_tokens_cap) {
        int new_cap = *raw_tokens_cap > 0 ? *raw_tokens_cap : 64;
        while (raw_need > new_cap) new_cap *= 2;
        int *tmp_raw = (int *)realloc(*raw_tokens, (size_t)new_cap * sizeof(int));
        if (!tmp_raw) return -1;
        *raw_tokens = tmp_raw;
        *raw_tokens_cap = new_cap;
    }
    if (carry > *stable_text_cap) {
        int new_cap = *stable_text_cap > 0 ? *stable_text_cap : 64;
        while (carry > new_cap) new_cap *= 2;
        int *tmp_stable = (int *)realloc(*stable_text_tokens, (size_t)new_cap * sizeof(int));
        if (!tmp_stable) return -1;
        *stable_text_tokens = tmp_stable;
        *stable_text_cap = new_cap;
    }

    int tail_off = n_emitted_text_tokens - carry;
    if (tail_off < 0) tail_off = 0;
    if (raw_lead) (*raw_tokens)[0] = QWEN_TOKEN_ASR_TEXT;
    if (carry > 0 && emitted_text_tokens) {
        memcpy((*raw_tokens) + raw_lead,
               emitted_text_tokens + tail_off,
               (size_t)carry * sizeof(int));
        memcpy(*stable_text_tokens,
               emitted_text_tokens + tail_off,
               (size_t)carry * sizeof(int));
    }

    *n_raw_tokens = raw_need;
    *n_stable_text_tokens = carry;
    return 0;
}

/* ========================================================================
 * Streaming Transcription (chunked rollback + encoder window cache)
 *
 * Decoder-side behavior follows the official streaming policy:
 * 1. Consume audio in fixed chunks (default 2 seconds).
 * 2. Use prefix rollback:
 *    - first N chunks: no text prefix,
 *    - later chunks: previous decoded tokens minus last K unfixed tokens.
 * 3. Decode only up to a bounded number of new tokens each step.
 * 4. Emit token deltas from the stable frontier.
 *
 * Encoder-side optimization:
 * - The encoder uses local attention windows, so completed windows are
 *   immutable.
 * - We cache completed window outputs once and only re-encode the current
 *   partial tail window.
 * - Decoder prefill still consumes all encoder tokens
 *   ([cached windows] + [current partial window]).
 * ======================================================================== */

/* Internal streaming implementation. When live!=NULL, audio is read
 * incrementally from the live buffer; when NULL, samples/n_samples
 * provide the complete audio upfront. */
static char *stream_impl(qwen_ctx_t *ctx, const float *samples, int n_samples,
                          qwen_live_audio_t *live) {
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int chunk_samples = (int)(ctx->stream_chunk_sec * QWEN_SAMPLE_RATE);
    int rollback = ctx->stream_rollback;
    int unfixed_chunks = ctx->stream_unfixed_chunks;
    int max_new_tokens = ctx->stream_max_new_tokens > 0 ? ctx->stream_max_new_tokens : 32;

    const float *audio_samples = samples;
    int64_t audio_n_samples = n_samples;
    float *compacted_samples = NULL;
    if (!live && ctx->skip_silence) {
        int compacted_n = n_samples;
        compacted_samples = compact_silence(samples, n_samples, &compacted_n);
        if (compacted_samples) audio_samples = compacted_samples;
        audio_n_samples = compacted_n;
        if (qwen_verbose >= 1) {
            float used_pct = 100.0f * (float)audio_n_samples /
                             (float)(n_samples > 0 ? n_samples : 1);
            float skipped_pct = 100.0f - used_pct;
            if (skipped_pct < 0.0f) skipped_pct = 0.0f;
            fprintf(stderr, "Silence skip: used %.1f%%, skipped %.1f%% (%d -> %lld samples)\n",
                    used_pct, skipped_pct, n_samples, (long long)audio_n_samples);
        }
    }

    /* For live mode, keep a local rolling buffer with global sample base. */
    float *local_samples = NULL;
    int64_t local_n_samples = 0;
    int64_t local_capacity = 0;
    int64_t local_base_sample = 0;
    int live_eof = 0;

    if (live) {
        /* Seed local buffer with whatever is available now. */
        pthread_mutex_lock(&live->mutex);
        int64_t live_start = live->sample_offset;
        int64_t live_count = live->n_samples;
        live_eof = live->eof;
        local_n_samples = live_count;
        local_base_sample = live_start;
        if (local_n_samples > 0) {
            local_capacity = local_n_samples + chunk_samples * 4;
            if ((uint64_t)local_capacity > (uint64_t)(SIZE_MAX / sizeof(float))) {
                pthread_mutex_unlock(&live->mutex);
                return NULL;
            }
            local_samples = (float *)malloc((size_t)local_capacity * sizeof(float));
            if (!local_samples) {
                pthread_mutex_unlock(&live->mutex);
                return NULL;
            }
            memcpy(local_samples, live->samples, (size_t)local_n_samples * sizeof(float));
        }
        /* Producer buffer is now mirrored locally: reset it to bound memory. */
        live->sample_offset = live_start + live_count;
        live->n_samples = 0;
        pthread_mutex_unlock(&live->mutex);
        audio_samples = local_samples;
        audio_n_samples = local_base_sample + local_n_samples;
    } else {
        if (audio_n_samples > INT_MAX) {
            free(compacted_samples);
            return NULL;
        }
        local_base_sample = 0;
        local_n_samples = audio_n_samples;
    }

    ctx->perf_total_ms = 0;
    ctx->perf_text_tokens = 0;
    ctx->perf_audio_ms = live ? 0.0 : 1000.0 * (double)n_samples / (double)QWEN_SAMPLE_RATE;
    ctx->perf_encode_ms = 0;
    ctx->perf_decode_ms = 0;
    int enc_window_frames = ctx->config.enc_n_window_infer;
    if (enc_window_frames < 100) enc_window_frames = 100;
    if (enc_window_frames > 800) enc_window_frames = 800;
    int enc_window_samples = enc_window_frames * QWEN_HOP_LENGTH;
    const char *no_draft_env = getenv("QWEN_STREAM_NO_DRAFT");
    int use_draft = !(no_draft_env && no_draft_env[0] != '\0' && strcmp(no_draft_env, "0") != 0);
    const char *no_agree_env = getenv("QWEN_STREAM_NO_AGREE");
    int use_agree = !(no_agree_env && no_agree_env[0] != '\0' && strcmp(no_agree_env, "0") != 0);
    const char *no_cache_env = getenv("QWEN_STREAM_NO_ENC_CACHE");
    int use_enc_cache = 1;
    if (no_cache_env && no_cache_env[0] != '\0' && strcmp(no_cache_env, "0") != 0) {
        use_enc_cache = 0;
    }
    if (live && !use_enc_cache) {
        if (qwen_verbose >= 1) {
            fprintf(stderr, "Streaming (live): forcing encoder cache on (no-cache mode disabled)\n");
        }
        use_enc_cache = 1;
    }

    /* Sliding-window limits for long streams: bound encoder tokens and
     * prefix tokens fed to the decoder so memory/compute stay flat.
     * 4 windows × 8 s = 32 s of audio context; 150 prefix tokens ≈
     * 140 text tokens of decoder context.  raw_tokens array itself
     * grows unbounded (negligible memory) for correct text matching. */
    #define QWEN_STREAM_MAX_ENC_WINDOWS  4
    #define QWEN_STREAM_MIN_PARTIAL_SAMPLES 800   /* 50 ms */
    #define QWEN_STREAM_TOKENS_PER_SEC 10.0
    #define QWEN_STREAM_MAX_PREFIX_TOKENS 150
    #define QWEN_STREAM_MAX_REPEAT_TOKEN_RUN 12
    #define QWEN_STREAM_OVERLAP_MAX_TOKENS 48
    #define QWEN_STREAM_OVERLAP_MIN_TOKENS 4
    #define QWEN_STREAM_DEGEN_MAX_PERIOD 32
    #define QWEN_STREAM_DEGEN_MIN_REPEATS 4
    /* Steps at the token limit without progress before the recovery reset. */
    #define QWEN_STREAM_STALE_CHUNKS 4
    /* Much longer and the 1.7B model drifts into writing no punctuation. */
    #define QWEN_STREAM_RESET_INTERVAL_SEC 90.0
    #define QWEN_STREAM_RESET_CARRY_TOKENS 24
    /* A periodic reset keeps this many completed encoder windows, so the
     * carried text keeps the audio it came from; without it the decoder
     * tends to end its turn on every chunk until the next reset. */
    #define QWEN_STREAM_RESET_KEEP_ENC_WINDOWS 1
    /* A pinned text prefix can make the decoder end its turn on every
     * chunk; after this many decode steps without a new token, drop it. */
    #define QWEN_STREAM_QUIET_RESET_CHUNKS 4
    /* A loop this soon after a recovery, within the window of audio the
     * recovery kept, drops the audio; until then the alignment after a
     * recovery may wait for more text. In seconds, rounded to chunks. */
    #define QWEN_STREAM_RECOVERY_WINDOW_SEC 8.0
    /* MIN_SCORE 6 is three net matching tokens. */
    #define QWEN_STREAM_STITCH_TAIL 48
    #define QWEN_STREAM_STITCH_MIN_SCORE 6
    /* Stream-mode silence skip: chunks whose RMS sits below an adaptive
     * threshold (P25 of recent chunk RMS × ratio) are skipped past the
     * encoder/decoder/emit work. Mirrors compact_silence's offline gating. */
    #define QWEN_STREAM_SILENCE_BASE_RMS 0.002f
    #define QWEN_STREAM_SILENCE_MAX_RMS 0.025f
    #define QWEN_STREAM_SILENCE_RATIO 1.8f
    #define QWEN_STREAM_RMS_HIST_LEN 16
    #define QWEN_STREAM_MAX_SILENCE_SKIPS 60
    /* Pause test for the flush and the pause reset. The chunk test above
     * calls the quieter part of continuous speech silent; this one compares
     * 30 ms frames with the noise floor, which in speech falls between words. */
    #define QWEN_STREAM_EP_FRAME 480
    #define QWEN_STREAM_EP_HIST 2000
    #define QWEN_STREAM_EP_MIN_HIST 100
    #define QWEN_STREAM_EP_FLOOR_PCT 10
    #define QWEN_STREAM_EP_SPEECH_RATIO 3.0f
    #define QWEN_STREAM_EP_MIN_SPEECH_RMS 0.001f
    #define QWEN_STREAM_EP_SEC 1.0

    if (qwen_verbose >= 2) {
        if (live)
            fprintf(stderr,
                    "Streaming (live): chunk=%.1f s, rollback=%d, "
                    "unfixed=%d, max_new=%d, enc_window=%.1fs, enc_cache=%s, prefix=%s, "
                    "max_enc_win=%d, max_prefix=%d\n",
                    ctx->stream_chunk_sec, rollback,
                    unfixed_chunks, max_new_tokens,
                    (float)enc_window_frames / 100.0f,
                    use_enc_cache ? "on" : "off",
                    ctx->past_text_conditioning ? "on" : "off",
                    QWEN_STREAM_MAX_ENC_WINDOWS, QWEN_STREAM_MAX_PREFIX_TOKENS);
        else
            fprintf(stderr,
                    "Streaming: %lld samples (%.1f s), chunk=%.1f s, rollback=%d, "
                    "unfixed=%d, max_new=%d, enc_window=%.1fs, enc_cache=%s, prefix=%s\n",
                    (long long)audio_n_samples, (float)audio_n_samples / QWEN_SAMPLE_RATE,
                    ctx->stream_chunk_sec, rollback,
                    unfixed_chunks, max_new_tokens,
                    (float)enc_window_frames / 100.0f,
                    use_enc_cache ? "on" : "off",
                    ctx->past_text_conditioning ? "on" : "off");
    }

    /* Load tokenizer */
    qwen_tokenizer_t *tokenizer = qwen_tokenizer_load(ctx->model_dir);
    if (!tokenizer) {
        free(compacted_samples);
        return NULL;
    }
    if (prepare_prompt_tokens(ctx, tokenizer) != 0) {
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }

    /* In non-interactive mode (no token callback) with pre-loaded audio,
     * streaming chunks are not externally consumed and the final answer is
     * already produced by a full refinement pass. Skip chunk-by-chunk
     * decoding entirely. (In live mode we must still use the chunked loop.) */
    if (!ctx->token_cb && !ctx->stream_update_cb && !live) {
        if (qwen_verbose >= 2) {
            fprintf(stderr, "Streaming: no token callback, using direct final refinement\n");
        }
        if (audio_n_samples > INT_MAX) {
            qwen_tokenizer_free(tokenizer);
            free(compacted_samples);
            return NULL;
        }
        char *text = transcribe_segment(ctx, audio_samples, (int)audio_n_samples,
                                        tokenizer, NULL, 0, NULL);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return text;
    }

    /* Raw decoded history (language + <asr_text> + text), tokenized. */
    int *raw_tokens = (int *)malloc(8192 * sizeof(int));
    int n_raw_tokens = 0;
    int raw_tokens_cap = 8192;

    /* Stable committed text tokens already emitted to stdout. */
    int *stable_text_tokens = (int *)malloc(8192 * sizeof(int));
    int n_stable_text_tokens = 0;
    int stable_text_cap = 8192;
    int *emitted_text_tokens = (int *)malloc(8192 * sizeof(int));
    int n_emitted_text_tokens = 0;
    int emitted_text_cap = 8192;
    int stagnant_chunks = 0;
    int quiet_chunks = 0;
    /* Loop break while decoding. The 1.7B model gets past a loop with its
     * runner-up token; the 0.6B model only varies the loop, so its step
     * ends instead. Five copies for 0.6B spare a spoken "はいはいはいはい". */
    int big_model = ctx->config.dec_hidden > 1024;
    int loop_break_reps = big_model ? 4 : 5;
    int loop_nudges = big_model ? 1 : 0;
    int loop_break_prev_chunk = 0;
    /* A restart that keeps audio re-decodes text already emitted; the next
     * text is aligned to it (stitch_pending), after a loop recovery once
     * there is enough of it (stitch_hold). */
    int stitch_pending = 0;
    int stitch_hold = 0;
    int stitch_tries = 0;
    int last_recovery_chunk = -1000;
    /* Raw tokens [0, min_prefix_raw) have been emitted, or after a restart
     * are a new decode of emitted text (the alignment skipped them). They
     * stay in the decoder prefix from then on: text already written out
     * cannot be revised, so the decoder must not be free to re-decode it. */
    int min_prefix_raw = 0;
    /* Raw tokens [0, agreed_raw) came out the same in two consecutive steps.
     * Only those are committed or fed back as prefix: a step that has heard
     * the first word of a sentence can guess the rest, and fed back, the
     * guess would confirm itself. */
    int agreed_raw = 0;
    int held_text_tokens = 0;        /* decoded on the last chunk, not yet emitted */
    int64_t idle_seen_samples = -1;  /* live sample count when last seen to grow */
    double idle_since_ms = 0;
    int64_t last_flush_samples = -1; /* live sample count at the last idle flush */
    /* Update callback state: bytes of a committed character still waiting
     * for its remaining bytes, and the tail last reported. */
    stream_str_t upd_commit = {0}, upd_tail = {0}, upd_last_tail = {0};
    char upd_pending[4];
    size_t upd_pending_len = 0;
    /* Result text accumulator */
    size_t result_cap = 4096;
    size_t result_len = 0;
    char *result = (char *)malloc(result_cap);
    if (!raw_tokens || !stable_text_tokens || !emitted_text_tokens || !result) {
        free(raw_tokens);
        free(stable_text_tokens);
        free(emitted_text_tokens);
        free(result);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }
    result[0] = '\0';

    /* Single-token decoder input buffer reused across all chunks. */
    float *tmp_embed = (float *)malloc(dim * sizeof(float));
    if (!tmp_embed) {
        free(raw_tokens);
        free(stable_text_tokens);
        free(emitted_text_tokens);
        free(result);
        qwen_tokenizer_free(tokenizer);
        free(compacted_samples);
        return NULL;
    }

    int chunk_idx = 0;
    int64_t audio_cursor = 0;
    /* --stream-json: resets and recoveries go to stderr, one line each,
     * stamped with the stream position. */
#define STREAM_EVENT(...) do { \
        if (ctx->stream_log) { \
            fprintf(stderr, "[stream %.1fs] ", (double)audio_cursor / QWEN_SAMPLE_RATE); \
            fprintf(stderr, __VA_ARGS__); \
            fputc('\n', stderr); \
        } \
    } while (0)
    char logged_lang[64] = "";
    int64_t behind_logged_at = -1;
    int64_t decoded_cursor = 0;   /* audio_cursor at the last decoded step */
    /* Audio before this has all its text emitted. A reset that drops held
     * text (to decode it again) does not move it. */
    int64_t emitted_cursor = 0;
    /* After a restart that keeps audio, the next step decodes that audio
     * again; it counts as new for the step's token limit. */
#define STREAM_REDECODE_FROM_KEPT() do { \
        decoded_cursor = (n_enc_cache > enc_cache_start) \
            ? enc_cache[enc_cache_start].start_sample \
            : (audio_cursor / enc_window_samples) * (int64_t)enc_window_samples; \
    } while (0)
    stream_enc_window_t *enc_cache = NULL;
    int n_enc_cache = 0;
    int enc_cache_start = 0;  /* first live entry (older ones evicted) */
    int enc_cache_cap = 0;
    int enc_cached_seq_total = 0;
    int64_t next_window_start = 0;
    /* Resets. Each drops the decoder's text state and some cached audio:
     *
     *   reset            text kept         windows kept   audio decoded again
     *   periodic trim    last 24 tokens    1              (with the text)
     *     no punctuation language header  1              yes, aligned
     *   stall            language header   none           partial window, aligned
     *   pause            none              none           unemitted part, aligned;
     *                                                     emitted part silenced
     *   loop, soft       language header   1              yes, aligned after a wait
     *   loop, hard       language header   none           no
     *
     * "Aligned": the new text is joined to the emitted text by alignment
     * instead of being emitted whole. */
#define STREAM_RESET_TEXT(keep_) do { \
        n_raw_tokens = (keep_); \
        n_stable_text_tokens = 0; \
        min_prefix_raw = (keep_); \
        agreed_raw = (keep_); \
        held_text_tokens = 0; \
        prev_prefill_len = 0; \
        period_held = 0; \
    } while (0)
    /* Negative windows_: clear the cache and restart the window grid at at_. */
#define STREAM_RESET_AUDIO(windows_, at_) do { \
        if ((windows_) < 0) \
            stream_clear_enc_cache(enc_cache, &n_enc_cache, &enc_cache_start, \
                                   &enc_cached_seq_total, &next_window_start, (at_)); \
        else \
            stream_trim_enc_cache(enc_cache, n_enc_cache, &enc_cache_start, \
                                  &enc_cached_seq_total, (windows_)); \
    } while (0)
#define STREAM_REALIGN(hold_) do { \
        stitch_pending = 1; \
        stitch_tries = 0; \
        stitch_hold = (hold_); \
        STREAM_REDECODE_FROM_KEPT(); \
    } while (0)
#define STREAM_RESET_DONE() do { \
        last_trim_chunk = chunk_idx; \
        seam_space = 1; \
    } while (0)
    float *prev_prefill_embeds = NULL;
    int prev_prefill_len = 0;
    int prev_prefill_cap = 0;
    int prefill_total_tokens = 0;
    int prefill_reused_tokens = 0;
    float rms_hist[QWEN_STREAM_RMS_HIST_LEN];
    int rms_hist_pos = 0;
    int rms_hist_count = 0;
    int consecutive_silence_skips = 0;
    float *ep_hist = (float *)malloc(QWEN_STREAM_EP_HIST * sizeof(float));
    float *ep_sorted = (float *)malloc(QWEN_STREAM_EP_HIST * sizeof(float));
    int ep_hist_pos = 0;
    int ep_hist_count = 0;
    double trail_sil_sec = 0;       /* audio at the end with no speech frame */
    int chunk_no_speech = 0;        /* the pause test found no speech in this chunk */
    int known_text_before = 0;
    /* The first text after a restart can lack its leading space. */
    int seam_space = 0;
    /* A flush held back the sentence mark that ended its text. */
    int period_held = 0;
    /* After a pause the text before it would keep shaping what follows, so
     * it is dropped; without --language the language header too, as the
     * speaker may have switched. Audio whose text is out is silenced before
     * it is encoded again: aligned against it, new sentences that repeat
     * earlier wording would be dropped. Audio decoded but not yet emitted is
     * decoded again and aligned. */
    double pause_reset_sec = ctx->stream_pause_reset_ms / 1000.0;
    double silent_run_sec = 0;
    int pause_reset_done = 0;
#define STREAM_PAUSE_RESET() do { \
        int realign_ = emitted_cursor < decoded_cursor; \
        STREAM_RESET_TEXT(0); \
        STREAM_RESET_AUDIO(0, 0); \
        if (local_samples) { \
            int64_t mf_ = next_window_start > local_base_sample ? next_window_start : local_base_sample; \
            int64_t mt_ = emitted_cursor; \
            if (mt_ > local_base_sample + local_n_samples) mt_ = local_base_sample + local_n_samples; \
            for (int64_t m_ = mf_; m_ < mt_; m_++) local_samples[m_ - local_base_sample] = 0.0f; \
        } \
        quiet_chunks = 0; \
        STREAM_REALIGN(0); \
        stitch_pending = realign_; \
        pause_reset_done = 1; \
        STREAM_RESET_DONE(); \
        if (qwen_verbose >= 2) fprintf(stderr, "  Pause reset applied\n"); \
        STREAM_EVENT("pause reset"); \
    } while (0)

    int reset_interval_chunks =
        (int)(QWEN_STREAM_RESET_INTERVAL_SEC / ctx->stream_chunk_sec + 0.5);
    if (reset_interval_chunks < 4) reset_interval_chunks = 4;
    int recovery_window_chunks = stream_sec_to_chunks(QWEN_STREAM_RECOVERY_WINDOW_SEC,
                                                      ctx->stream_chunk_sec);
    /* The periodic trim counts from the last reset of any kind. */
    int last_trim_chunk = -1;

    while (audio_cursor < audio_n_samples || (live && !live_eof)) {
        int is_flush = 0;
        /* Live mode: wait until we have enough data for the next chunk. */
        if (live) {
            int64_t want = audio_cursor + chunk_samples;
            pthread_mutex_lock(&live->mutex);
            while (live->sample_offset + live->n_samples < want && !live->eof) {
                /* Idle flush: a live source that stops sending (a speaker
                 * pausing behind a voice gate) would otherwise leave the
                 * partial chunk and the held-back tail undecoded until audio
                 * resumes. After stream_idle_flush_ms without new samples,
                 * decode what is buffered and emit everything. */
                int64_t have = live->sample_offset + live->n_samples;
                int pending = (held_text_tokens > 0 || have > audio_cursor) &&
                              have != last_flush_samples;
                /* A sentence mark held back by the last flush is confirmed
                 * when the pause has lasted as long as a pause reset needs,
                 * before that reset. */
                int confirm = !pending && period_held && held_text_tokens > 0;
                double now_ms = get_time_ms();
                if (have != idle_seen_samples) {
                    idle_seen_samples = have;
                    idle_since_ms = now_ms;
                }
                /* First the idle flush; once nothing is pending, the pause
                 * reset if the stall goes on. */
                double deadline_ms = -1;
                if (ctx->stream_idle_flush_ms > 0 && pending) {
                    deadline_ms = idle_since_ms + ctx->stream_idle_flush_ms;
                } else if (confirm) {
                    deadline_ms = idle_since_ms + (pause_reset_sec > 0 ? pause_reset_sec * 1000.0
                                                   : 2.0 * ctx->stream_idle_flush_ms);
                } else if (!pending && !pause_reset_done && pause_reset_sec > 0 &&
                           (n_raw_tokens > 0 || n_enc_cache > enc_cache_start)) {
                    deadline_ms = idle_since_ms + pause_reset_sec * 1000.0;
                }
                if (deadline_ms < 0) {
                    pthread_cond_wait(&live->cond, &live->mutex);
                    continue;
                }
                double remain_ms = deadline_ms - now_ms;
                if (remain_ms <= 0) {
                    if (pending || confirm) {
                        is_flush = 1;
                        last_flush_samples = have;
                        break;
                    }
                    STREAM_PAUSE_RESET();
                    continue;
                }
                struct timeval tv;
                gettimeofday(&tv, NULL);
                int64_t wake_us = (int64_t)tv.tv_sec * 1000000 + tv.tv_usec +
                                  (int64_t)(remain_ms * 1000.0) + 1000;
                struct timespec ts;
                ts.tv_sec = (time_t)(wake_us / 1000000);
                ts.tv_nsec = (long)(wake_us % 1000000) * 1000;
                pthread_cond_timedwait(&live->cond, &live->mutex, &ts);
            }

            int64_t live_start = live->sample_offset;
            int64_t live_count = live->n_samples;
            int64_t live_end = live_start + live_count;
            int is_eof_now = live->eof;

            int64_t local_end = local_base_sample + local_n_samples;
            if (local_end < live_start) {
                if (qwen_verbose >= 1) {
                    fprintf(stderr,
                            "Streaming (live): local buffer overrun, resyncing "
                            "(local_end=%lld, live_start=%lld)\n",
                            (long long)local_end, (long long)live_start);
                }
                local_base_sample = live_start;
                local_n_samples = 0;
                local_end = local_base_sample;
            }

            if (live_end > local_end) {
                int64_t delta64 = live_end - local_end;
                int64_t src_off64 = local_end - live_start;
                if (delta64 < 0 || src_off64 < 0 || src_off64 > live_count) {
                    pthread_mutex_unlock(&live->mutex);
                    break;
                }

                if (local_n_samples + delta64 > local_capacity) {
                    int64_t new_cap = local_capacity > 0 ? local_capacity : 32000;
                    while (new_cap < local_n_samples + delta64) new_cap *= 2;
                    if ((uint64_t)new_cap > (uint64_t)(SIZE_MAX / sizeof(float))) {
                        pthread_mutex_unlock(&live->mutex);
                        break;
                    }
                    float *tmp = (float *)realloc(local_samples,
                                                  (size_t)new_cap * sizeof(float));
                    if (!tmp) {
                        pthread_mutex_unlock(&live->mutex);
                        break;
                    }
                    local_samples = tmp;
                    local_capacity = new_cap;
                }
                memcpy(local_samples + (size_t)local_n_samples,
                       live->samples + (size_t)src_off64,
                       (size_t)delta64 * sizeof(float));
                local_n_samples += delta64;
            }

            /* Producer buffer is mirrored locally: reset it to bound memory. */
            live->sample_offset = live_end;
            live->n_samples = 0;
            live_eof = is_eof_now;
            pthread_mutex_unlock(&live->mutex);

            audio_samples = local_samples;
            audio_n_samples = local_base_sample + local_n_samples;
            ctx->perf_audio_ms = 1000.0 * (double)audio_n_samples / (double)QWEN_SAMPLE_RATE;
        }

        double chunk_t0 = get_time_ms();
        if (live && ctx->stream_log) {
            double behind = (double)(audio_n_samples - audio_cursor) / QWEN_SAMPLE_RATE;
            if (behind > 2.0 * ctx->stream_chunk_sec &&
                (behind_logged_at < 0 || audio_cursor - behind_logged_at >= 30 * QWEN_SAMPLE_RATE)) {
                STREAM_EVENT("%.1fs of input not decoded yet", behind);
                behind_logged_at = audio_cursor;
            }
        }
        if (is_flush && audio_n_samples - audio_cursor < QWEN_SAMPLE_RATE / 5 &&
            audio_cursor > 0) {
            /* Under 200 ms of new audio: re-decode the same span to release
             * the held tail and leave the sliver for the next chunk. */
        } else {
            audio_cursor += chunk_samples;
            if (audio_cursor > audio_n_samples) audio_cursor = audio_n_samples;
        }
        int is_final = live ? (live_eof && audio_cursor >= audio_n_samples)
                            : (audio_cursor >= audio_n_samples);

        chunk_no_speech = 0;
        if (!is_final && !is_flush) {
            int64_t chunk_start = audio_cursor - chunk_samples;
            if (chunk_start < 0) chunk_start = 0;
            int64_t chunk_local_off = chunk_start - local_base_sample;
            int64_t chunk_len = audio_cursor - chunk_start;

            if (chunk_local_off >= 0 &&
                chunk_local_off + chunk_len <= local_n_samples &&
                chunk_len > 0) {
                /* Pause test, run on every chunk: the stall reset needs to
                 * know whether a chunk had speech at all. */
                int had_speech = 0;
                int last_speech = -1;
                int run = 0;
                double gap_max = 0;           /* longest pause that ended in this chunk */
                double prev_trail = trail_sil_sec;
                int n_fr = (int)(chunk_len / QWEN_STREAM_EP_FRAME);
                float sp_thr = 0;
                if (ep_hist && ep_sorted && ep_hist_count >= QWEN_STREAM_EP_MIN_HIST) {
                    memcpy(ep_sorted, ep_hist, (size_t)ep_hist_count * sizeof(float));
                    qsort(ep_sorted, (size_t)ep_hist_count, sizeof(float), cmp_float_asc);
                    sp_thr = ep_sorted[ep_hist_count * QWEN_STREAM_EP_FLOOR_PCT / 100] *
                             QWEN_STREAM_EP_SPEECH_RATIO;
                    if (sp_thr < QWEN_STREAM_EP_MIN_SPEECH_RMS) sp_thr = QWEN_STREAM_EP_MIN_SPEECH_RMS;
                }
                for (int f = 0; f < n_fr && ep_hist; f++) {
                    const float *fp = audio_samples + chunk_local_off + (int64_t)f * QWEN_STREAM_EP_FRAME;
                    float fe = 0.0f;
                    for (int k = 0; k < QWEN_STREAM_EP_FRAME; k++) fe += fp[k] * fp[k];
                    float fr = sqrtf(fe / QWEN_STREAM_EP_FRAME);
                    if (sp_thr > 0 && fr > sp_thr) {
                        double g = (double)run * QWEN_STREAM_EP_FRAME / QWEN_SAMPLE_RATE;
                        if (last_speech < 0) g += prev_trail;
                        if (g > gap_max) gap_max = g;
                        last_speech = f;
                        run = 0;
                    } else {
                        run++;
                    }
                    ep_hist[ep_hist_pos] = fr;
                    ep_hist_pos = (ep_hist_pos + 1) % QWEN_STREAM_EP_HIST;
                    if (ep_hist_count < QWEN_STREAM_EP_HIST) ep_hist_count++;
                }
                if (sp_thr <= 0) {
                    trail_sil_sec = 0;
                } else if (last_speech >= 0) {
                    had_speech = 1;
                    trail_sil_sec = (double)(n_fr - 1 - last_speech) * QWEN_STREAM_EP_FRAME / QWEN_SAMPLE_RATE;
                } else {
                    trail_sil_sec += (double)chunk_len / QWEN_SAMPLE_RATE;
                }
                chunk_no_speech = (sp_thr > 0 && !had_speech);

                /* Stream-mode silence skip: when --skip-silence is on, drop
                 * the encoder/decoder/emit work for chunks whose RMS sits
                 * below an adaptive threshold computed from recent chunk RMS
                 * values (P25 × ratio, mirrors compact_silence). Samples
                 * remain in the local buffer so future complete encoder
                 * windows still see them intact -- only the per-chunk emit
                 * cycle is short-circuited. Capped at
                 * QWEN_STREAM_MAX_SILENCE_SKIPS in a row so any pending
                 * decoder/text state still gets flushed during very long
                 * pauses. The flush and the pause reset hang off the same
                 * live, skip-silence path. */
                if (ctx->skip_silence && live) {
                    float energy = 0.0f;
                    for (int64_t i = 0; i < chunk_len; i++) {
                        float v = audio_samples[chunk_local_off + i];
                        energy += v * v;
                    }
                    float chunk_rms = sqrtf(energy / (float)chunk_len);

                    rms_hist[rms_hist_pos] = chunk_rms;
                    rms_hist_pos = (rms_hist_pos + 1) % QWEN_STREAM_RMS_HIST_LEN;
                    if (rms_hist_count < QWEN_STREAM_RMS_HIST_LEN) rms_hist_count++;

                    float threshold;
                    if (rms_hist_count < 4) {
                        threshold = QWEN_STREAM_SILENCE_BASE_RMS;
                    } else {
                        float sorted[QWEN_STREAM_RMS_HIST_LEN];
                        memcpy(sorted, rms_hist,
                               (size_t)rms_hist_count * sizeof(float));
                        qsort(sorted, (size_t)rms_hist_count, sizeof(float),
                              cmp_float_asc);
                        float p25 = sorted[rms_hist_count / 4];
                        threshold = p25 * QWEN_STREAM_SILENCE_RATIO;
                        if (threshold < QWEN_STREAM_SILENCE_BASE_RMS) {
                            threshold = QWEN_STREAM_SILENCE_BASE_RMS;
                        }
                        if (threshold > QWEN_STREAM_SILENCE_MAX_RMS) {
                            threshold = QWEN_STREAM_SILENCE_MAX_RMS;
                        }
                    }

                    /* A pause that fell between two chunk boundaries is seen
                     * only once speech resumes; reset before decoding it. */
                    if (had_speech && gap_max >= pause_reset_sec && pause_reset_sec > 0 &&
                        !pause_reset_done && held_text_tokens == 0 &&
                        (n_raw_tokens > 0 || n_enc_cache > enc_cache_start)) {
                        STREAM_PAUSE_RESET();
                    }
                    if (had_speech) pause_reset_done = 0;
                    silent_run_sec = trail_sil_sec;
                    if (qwen_verbose >= 2 && sp_thr > 0)
                        fprintf(stderr, "  Pause test: trailing %.2fs (speech > %.5f)\n",
                                trail_sil_sec, sp_thr);

                    if (trail_sil_sec >= QWEN_STREAM_EP_SEC && held_text_tokens > 0 &&
                        ctx->stream_idle_flush_ms > 0) {
                        /* The speaker has paused: decode now so the held tail
                         * comes out, not when speech resumes. */
                        is_flush = 1;
                    } else if (chunk_rms < threshold &&
                        consecutive_silence_skips < QWEN_STREAM_MAX_SILENCE_SKIPS) {
                        consecutive_silence_skips++;
                        if (qwen_verbose >= 2) {
                            fprintf(stderr,
                                    "  Silence skip: rms=%.5f thresh=%.5f "
                                    "(chunk %d, run=%d)\n",
                                    chunk_rms, threshold, chunk_idx,
                                    consecutive_silence_skips);
                        }
                        if (!pause_reset_done && pause_reset_sec > 0 && held_text_tokens == 0 &&
                            silent_run_sec >= pause_reset_sec &&
                            (n_raw_tokens > 0 || n_enc_cache > enc_cache_start)) {
                            STREAM_PAUSE_RESET();
                        }
                        ctx->perf_total_ms += get_time_ms() - chunk_t0;
                        chunk_idx++;
                        continue;
                    }
                    if (!is_flush) consecutive_silence_skips = 0;
                }
            }
        }

        if (!is_flush && !(ctx->skip_silence && live && !is_final)) {
            silent_run_sec = 0;
            pause_reset_done = 0;
        }

        /* Encoder path:
         * - default: cache completed local-attention windows and re-encode only
         *   the current partial tail window,
         * - debug fallback (`QWEN_STREAM_NO_ENC_CACHE=1`): re-encode full audio
         *   prefix every chunk. */
        double t0 = get_time_ms();
        int enc_seq_len = 0;
        float *enc_output = NULL;
        int64_t full_end = (audio_cursor / enc_window_samples) * (int64_t)enc_window_samples;
        known_text_before = n_emitted_text_tokens + held_text_tokens;

        if (!use_enc_cache) {
            if (audio_cursor > INT_MAX) {
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            if (stream_encode_span(ctx, audio_samples, (int)audio_cursor,
                                   &enc_output, &enc_seq_len) != 0 ||
                !enc_output || enc_seq_len <= 0) {
                free(enc_output);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            double enc_ms = get_time_ms() - t0;
            ctx->perf_encode_ms += enc_ms;
            if (qwen_verbose >= 2) {
                fprintf(stderr,
                        "  Encoder: %d tokens from 0.0-%.1f s (full recompute, %.0f ms)\n",
                        enc_seq_len,
                        (float)audio_cursor / QWEN_SAMPLE_RATE,
                        enc_ms);
            }
        } else {
            int enc_failed = 0;

            while (next_window_start < full_end) {
                int64_t ws = next_window_start;
                int64_t ws_local_off = ws - local_base_sample;
                if (ws_local_off < 0 ||
                    ws_local_off + enc_window_samples > local_n_samples) {
                    enc_failed = 1;
                    break;
                }
                float *win_enc = NULL;
                int win_seq = 0;
                if (stream_encode_span(ctx,
                                       audio_samples + (size_t)ws_local_off,
                                       enc_window_samples,
                                       &win_enc, &win_seq) != 0 ||
                    !win_enc || win_seq <= 0) {
                    free(win_enc);
                    enc_failed = 1;
                    break;
                }

                if (n_enc_cache == enc_cache_cap) {
                    int new_cap = enc_cache_cap > 0 ? enc_cache_cap * 2 : 8;
                    stream_enc_window_t *tmp = (stream_enc_window_t *)realloc(
                        enc_cache, (size_t)new_cap * sizeof(stream_enc_window_t));
                    if (!tmp) {
                        free(win_enc);
                        enc_failed = 1;
                        break;
                    }
                    enc_cache = tmp;
                    enc_cache_cap = new_cap;
                }

                enc_cache[n_enc_cache].start_sample = ws;
                enc_cache[n_enc_cache].n_samples = enc_window_samples;
                enc_cache[n_enc_cache].seq_len = win_seq;
                enc_cache[n_enc_cache].enc_output = win_enc;
                n_enc_cache++;
                enc_cached_seq_total += win_seq;
                next_window_start += enc_window_samples;
            }

            float *partial_enc = NULL;
            int partial_seq = 0;
            /* A few samples past a window boundary are too short to encode
             * and hold no speech. */
            if (!enc_failed && full_end < audio_cursor &&
                audio_cursor - full_end >= QWEN_STREAM_MIN_PARTIAL_SAMPLES) {
                int64_t partial_samples64 = audio_cursor - full_end;
                int64_t partial_off64 = full_end - local_base_sample;
                if (partial_samples64 > INT_MAX || partial_off64 < 0 ||
                    partial_off64 + partial_samples64 > local_n_samples) {
                    enc_failed = 1;
                } else if (stream_encode_span(ctx,
                                       audio_samples + (size_t)partial_off64,
                                       (int)partial_samples64,
                                       &partial_enc, &partial_seq) != 0) {
                    free(partial_enc);
                    partial_enc = NULL;
                    enc_failed = 1;
                }
            }

            if (enc_failed) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            /* Evict old encoder windows beyond the sliding-window limit
             * to keep decoder sequence length (and KV cache) bounded. */
            {
                int evicted = stream_trim_enc_cache(enc_cache, n_enc_cache,
                                                    &enc_cache_start,
                                                    &enc_cached_seq_total,
                                                    QWEN_STREAM_MAX_ENC_WINDOWS);
                if (evicted && qwen_monitor) {
                    fprintf(stderr, "\xe2\x9f\xb3");  /* ⟳ = window eviction */
                    fflush(stderr);
                }
            }

            enc_seq_len = enc_cached_seq_total + partial_seq;
            if (enc_seq_len <= 0) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            enc_output = (float *)malloc((size_t)enc_seq_len * dim * sizeof(float));
            if (!enc_output) {
                free(partial_enc);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }

            int enc_off = 0;
            for (int i = enc_cache_start; i < n_enc_cache; i++) {
                memcpy(enc_output + (size_t)enc_off * dim,
                       enc_cache[i].enc_output,
                       (size_t)enc_cache[i].seq_len * dim * sizeof(float));
                enc_off += enc_cache[i].seq_len;
            }
            if (partial_seq > 0 && partial_enc) {
                memcpy(enc_output + (size_t)enc_off * dim,
                       partial_enc, (size_t)partial_seq * dim * sizeof(float));
            }
            free(partial_enc);

            if (qwen_verbose >= 2) {
                double enc_ms = get_time_ms() - t0;
                ctx->perf_encode_ms += enc_ms;
                fprintf(stderr,
                        "  Encoder: %d tokens from 0.0-%.1f s (cached windows=%d, partial=%.1f s, %.0f ms)\n",
                        enc_seq_len,
                        (float)audio_cursor / QWEN_SAMPLE_RATE,
                        n_enc_cache - enc_cache_start,
                        (float)(audio_cursor - full_end) / QWEN_SAMPLE_RATE,
                        enc_ms);
            }
            if (qwen_verbose < 2) {
                double enc_ms = get_time_ms() - t0;
                ctx->perf_encode_ms += enc_ms;
            }
            if (qwen_monitor) {
                fprintf(stderr, "\xe2\x96\xb6");  /* ▶ = encoder */
                fflush(stderr);
            }
        }

        /* Prefix rollback state:
         * we feed previously decoded raw tokens minus last `rollback` tokens.
         * This mirrors official streaming and keeps boundary text stable. */
        int n_prefix_tokens_full = 0;
        int n_prefix_tokens = 0;
        int prefix_offset = 0;
        int warmed_up = (chunk_idx >= unfixed_chunks || min_prefix_raw > 0);
        if (ctx->past_text_conditioning && warmed_up && n_raw_tokens > 0) {
            n_prefix_tokens_full = n_raw_tokens - rollback;
            if (use_agree && n_prefix_tokens_full > agreed_raw) n_prefix_tokens_full = agreed_raw;
            if (n_prefix_tokens_full < min_prefix_raw) n_prefix_tokens_full = min_prefix_raw;
            if (n_prefix_tokens_full > n_raw_tokens) n_prefix_tokens_full = n_raw_tokens;
            if (n_prefix_tokens_full < 0) n_prefix_tokens_full = 0;
            n_prefix_tokens = n_prefix_tokens_full;
            if (n_prefix_tokens > QWEN_STREAM_MAX_PREFIX_TOKENS) {
                n_prefix_tokens = QWEN_STREAM_MAX_PREFIX_TOKENS;
                prefix_offset = n_prefix_tokens_full - n_prefix_tokens;
            }
        }

        /* ---- Build input embeddings ---- */
        /* [PREFIX_HEAD] [prompt] [PREFIX_TAIL] [audio] [SUFFIX_BASE] [force-lang] [prefix_tokens] */
        int prefix_len = PREFIX_HEAD_LEN + ctx->n_prompt_tokens + PREFIX_TAIL_LEN;
        int suffix_len = SUFFIX_BASE_LEN + ctx->n_force_prompt_tokens;
        int total_seq = prefix_len + enc_seq_len + suffix_len + n_prefix_tokens;
        float *input_embeds = (float *)malloc((size_t)total_seq * dim * sizeof(float));
        if (!input_embeds) {
            free(enc_output);
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }

        int off = 0;
        for (int i = 0; i < PREFIX_HEAD_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  PROMPT_PREFIX_HEAD[i], dim);
            off++;
        }
        for (int i = 0; i < ctx->n_prompt_tokens; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  ctx->prompt_tokens[i], dim);
            off++;
        }
        for (int i = 0; i < PREFIX_TAIL_LEN; i++) {
            tok_embed_bf16_to_f32(input_embeds + off * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  PROMPT_PREFIX_TAIL[i], dim);
            off++;
        }

        for (int i = 0; i < enc_seq_len; i++)
            memcpy(input_embeds + (prefix_len + i) * dim,
                   enc_output + i * dim, dim * sizeof(float));
        free(enc_output);
        enc_output = NULL;

        int suffix_off = prefix_len + enc_seq_len;
        for (int i = 0; i < SUFFIX_BASE_LEN; i++)
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  PROMPT_SUFFIX_BASE[i], dim);

        for (int i = 0; i < ctx->n_force_prompt_tokens; i++)
            tok_embed_bf16_to_f32(input_embeds + (suffix_off + SUFFIX_BASE_LEN + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  ctx->force_prompt_tokens[i], dim);

        int text_off = suffix_off + suffix_len;
        for (int i = 0; i < n_prefix_tokens; i++)
            tok_embed_bf16_to_f32(input_embeds + (text_off + i) * dim,
                                  ctx->decoder.tok_embeddings_bf16,
                                  raw_tokens[prefix_offset + i], dim);

        /* ---- Decoder prefill + first token ---- */
        t0 = get_time_ms();
        int prefill_len = total_seq - 1;
        int reused_prefill = 0;
        if (prev_prefill_embeds && prev_prefill_len > 0) {
            int cmp_len = prefill_len < prev_prefill_len ? prefill_len : prev_prefill_len;
            size_t row_bytes = (size_t)dim * sizeof(float);
            while (reused_prefill < cmp_len) {
                const float *a = prev_prefill_embeds + (size_t)reused_prefill * dim;
                const float *b = input_embeds + (size_t)reused_prefill * dim;
                if (memcmp(a, b, row_bytes) != 0) break;
                reused_prefill++;
            }
        }
        /* Decoder KV reuse:
         * keep the longest unchanged prefill prefix and only prefill delta tokens. */
        ctx->kv_cache_len = reused_prefill;
        int delta_prefill = prefill_len - reused_prefill;
        if (delta_prefill > 0) {
            qwen_decoder_prefill(ctx,
                                 input_embeds + (size_t)reused_prefill * dim,
                                 delta_prefill);
        }
        prefill_total_tokens += prefill_len;
        prefill_reused_tokens += reused_prefill;

        float *last_embed = input_embeds + (size_t)prefill_len * dim;
        int token = qwen_decoder_forward(ctx, last_embed);

        if (prefill_len > prev_prefill_cap) {
            int new_cap = prev_prefill_cap > 0 ? prev_prefill_cap : 64;
            while (new_cap < prefill_len) new_cap *= 2;
            float *tmp_prev = (float *)realloc(prev_prefill_embeds,
                                               (size_t)new_cap * dim * sizeof(float));
            if (tmp_prev) {
                prev_prefill_embeds = tmp_prev;
                prev_prefill_cap = new_cap;
            } else {
                prev_prefill_len = 0;
            }
        }
        if (prev_prefill_embeds && prev_prefill_cap >= prefill_len) {
            memcpy(prev_prefill_embeds, input_embeds,
                   (size_t)prefill_len * dim * sizeof(float));
            prev_prefill_len = prefill_len;
        } else {
            prev_prefill_len = 0;
        }
        free(input_embeds);

        double prefill_ms = get_time_ms() - t0;
        ctx->perf_decode_ms += prefill_ms;
        if (qwen_verbose >= 2)
            fprintf(stderr, "  Prefill: %d tokens (%d prefix, reused %d) (%.0f ms)\n",
                    total_seq, n_prefix_tokens, reused_prefill, prefill_ms);
        if (qwen_monitor) {
            fprintf(stderr, "\xc2\xb7");  /* · = prefill */
            fflush(stderr);
        }

        /* ---- Autoregressive decode ---- */
        t0 = get_time_ms();
        int n_generated = 0;

        /* After skipped chunks or a restart that keeps audio, a step covers
         * more than one chunk. */
        int step_cap = max_new_tokens;
        {
            double new_sec = (double)(audio_cursor - decoded_cursor) / QWEN_SAMPLE_RATE;
            int need = (int)(QWEN_STREAM_TOKENS_PER_SEC * new_sec + 0.5) + rollback;
            if (need > step_cap) step_cap = need;
        }
        decoded_cursor = audio_cursor;

        /* Collect ALL generated tokens (including language, <asr_text>, etc.) */
        int *chunk_tokens = (int *)malloc((size_t)step_cap * sizeof(int));
        if (!chunk_tokens) {
            ctx->perf_total_ms += get_time_ms() - chunk_t0;
            chunk_idx++;
            continue;
        }
        int n_chunk_tokens = 0;
        int loop_breaks = 0;
        int force_recovery = 0;

        /* The previous step already decoded the tokens after this prefix. If the
         * first new token matches, run them as one batch; steps that feed them
         * take their prediction from it instead of a single-token forward. */
        const int *draft = raw_tokens + n_prefix_tokens_full;
        int n_draft = n_raw_tokens - n_prefix_tokens_full;
        if (n_draft > QWEN_MULTI_MAX_X) n_draft = QWEN_MULTI_MAX_X;
        if (n_draft > step_cap) n_draft = step_cap;
        int draft_pred[QWEN_MULTI_MAX_X];
        int draft_used = 0, draft_live = 0;
        int draft_kv_base = ctx->kv_cache_len;
        if (use_draft && n_draft >= 2 && token == draft[0]) {
            float *draft_embeds = (float *)malloc((size_t)n_draft * dim * sizeof(float));
            if (draft_embeds) {
                for (int i = 0; i < n_draft; i++)
                    tok_embed_bf16_to_f32(draft_embeds + (size_t)i * dim,
                                          ctx->decoder.tok_embeddings_bf16, draft[i], dim);
                draft_live = qwen_decoder_forward_batch(ctx, draft_embeds, n_draft,
                                                        draft_pred) == 0;
                free(draft_embeds);
            }
            ctx->kv_cache_len = draft_kv_base;
        }

        while (n_generated < step_cap) {
            n_generated++;
            if (token == QWEN_TOKEN_ENDOFTEXT || token == QWEN_TOKEN_IM_END) break;

            chunk_tokens[n_chunk_tokens++] = token;

            if (draft_live && draft_used < n_draft && token == draft[draft_used]) {
                ctx->kv_cache_len = draft_kv_base + draft_used + 1;
                qwen_decoder_select_batch_row(draft_used);
                token = draft_pred[draft_used++];
            } else {
                draft_live = 0;
                tok_embed_bf16_to_f32(tmp_embed, ctx->decoder.tok_embeddings_bf16, token, dim);
                token = qwen_decoder_forward(ctx, tmp_embed);
            }

            if (token != QWEN_TOKEN_ENDOFTEXT && token != QWEN_TOKEN_IM_END) {
                int lb[64];
                int ln = 0;
                int from_prefix = 64 - n_chunk_tokens;
                if (from_prefix > n_prefix_tokens_full) from_prefix = n_prefix_tokens_full;
                for (int q = n_prefix_tokens_full - from_prefix; q < n_prefix_tokens_full && q >= 0; q++)
                    lb[ln++] = raw_tokens[q];
                int from_chunk = n_chunk_tokens < 64 - ln ? n_chunk_tokens : 64 - ln;
                for (int q = n_chunk_tokens - from_chunk; q < n_chunk_tokens; q++)
                    lb[ln++] = chunk_tokens[q];
                int lp = 0;
                int lr = stream_tail_repeat_blocks(lb, ln, 16, &lp);
                if (lp > 0 && lr >= (lp == 1 ? 2 * loop_break_reps : loop_break_reps) &&
                    token == lb[ln - lp]) {
                    if (loop_breaks >= loop_nudges || loop_break_prev_chunk) {
                        /* The recovery reset takes over. */
                        force_recovery = 1;
                        break;
                    }
                    token = qwen_decoder_next_excluding(ctx, token);
                    loop_breaks++;
                }
            }
        }
        if (n_draft >= 2 && qwen_verbose >= 2)
            fprintf(stderr, "  Draft: %d of %d previous tokens reused\n", draft_used, n_draft);
        if (loop_breaks > 0 && qwen_verbose >= 2)
            fprintf(stderr, "  Loop reset applied (%d)\n", loop_breaks);
        if (loop_breaks > 0 && !force_recovery)
            STREAM_EVENT("loop broken with the runner-up token");
        loop_break_prev_chunk = (loop_breaks > 0 && !force_recovery);

        double decode_ms = get_time_ms() - t0;
        ctx->perf_decode_ms += decode_ms;
        if (qwen_verbose >= 2)
            fprintf(stderr, "  Decode: %d tokens (%.0f ms, %.1f ms/token%s)\n",
                    n_generated, decode_ms,
                    n_generated > 0 ? decode_ms / n_generated : 0,
                    (n_generated >= step_cap &&
                     token != QWEN_TOKEN_ENDOFTEXT &&
                     token != QWEN_TOKEN_IM_END) ? ", hit max_new" : "");
        if (qwen_monitor) {
            /* ▪ = normal decode, ▸ = slow (>30ms/token) */
            double ms_per_tok = n_generated > 0 ? decode_ms / n_generated : 0;
            fprintf(stderr, "%s", ms_per_tok > 30 ? "\xe2\x96\xb8" : "\xe2\x96\xaa");
            fflush(stderr);
        }

        /* Update raw token history = full prefix + newly generated continuation.
         * Uses n_prefix_tokens_full (uncapped) so raw_tokens keeps the complete
         * token sequence for correct text-level matching in the commit phase. */
        int dropped_repeat_tokens = 0;
        if (n_chunk_tokens > 0) {
            int prev_tok = -1;
            int prev_run = 0;
            if (n_prefix_tokens_full > 0) {
                prev_tok = raw_tokens[n_prefix_tokens_full - 1];
                prev_run = 1;
                for (int j = n_prefix_tokens_full - 2; j >= 0; j--) {
                    if (raw_tokens[j] != prev_tok) break;
                    prev_run++;
                    if (prev_run >= QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) break;
                }
            }

            int out = 0;
            for (int i = 0; i < n_chunk_tokens; i++) {
                int tok = chunk_tokens[i];
                int suppress = 0;

                if (tok == prev_tok) {
                    prev_run++;
                    if (prev_run > QWEN_STREAM_MAX_REPEAT_TOKEN_RUN) suppress = 1;
                } else {
                    prev_tok = tok;
                    prev_run = 1;
                }

                if (suppress) {
                    dropped_repeat_tokens++;
                    continue;
                }
                chunk_tokens[out++] = tok;
            }
            n_chunk_tokens = out;
        }

        int n_agree = 0;
        while (n_agree < n_chunk_tokens && n_prefix_tokens_full + n_agree < n_raw_tokens &&
               raw_tokens[n_prefix_tokens_full + n_agree] == chunk_tokens[n_agree])
            n_agree++;
        int n_raw_new = n_prefix_tokens_full + n_chunk_tokens;
        /* A stall is a pinned text prefix keeping the decoder silent while
         * someone speaks. Chunks without speech neither count nor break it;
         * without past text there is no prefix to blame. */
        if (!chunk_no_speech) {
            if (!is_final && ctx->past_text_conditioning &&
                n_raw_new > 0 && n_raw_new <= n_raw_tokens)
                quiet_chunks++;
            else
                quiet_chunks = 0;
        }
        int quiet_reset = (warmed_up && quiet_chunks >= QWEN_STREAM_QUIET_RESET_CHUNKS);
        if (n_raw_new > raw_tokens_cap) {
            while (n_raw_new > raw_tokens_cap) raw_tokens_cap *= 2;
            int *tmp_raw = (int *)realloc(raw_tokens, (size_t)raw_tokens_cap * sizeof(int));
            if (!tmp_raw) {
                free(chunk_tokens);
                ctx->perf_total_ms += get_time_ms() - chunk_t0;
                chunk_idx++;
                continue;
            }
            raw_tokens = tmp_raw;
        }
        if (n_chunk_tokens > 0) {
            memcpy(raw_tokens + n_prefix_tokens_full, chunk_tokens,
                   (size_t)n_chunk_tokens * sizeof(int));
        }
        n_raw_tokens = n_raw_new;
        agreed_raw = n_prefix_tokens_full + n_agree;
        free(chunk_tokens);
        if (dropped_repeat_tokens > 0 && qwen_verbose >= 2) {
            fprintf(stderr, "  Decode: dropped %d repeated tokens\n", dropped_repeat_tokens);
        }

        /* Parse text region from raw stream output:
         * - default: language ... <asr_text> TEXT
         * - forced language: prompt already anchors language, so generated stream is TEXT. */
        int text_start = 0;
        if (ctx->n_force_prompt_tokens <= 0) {
            int asr_text_pos = -1;
            for (int i = 0; i < n_raw_tokens; i++) {
                if (raw_tokens[i] == QWEN_TOKEN_ASR_TEXT) {
                    asr_text_pos = i;
                    break;
                }
            }
            text_start = (asr_text_pos >= 0) ? asr_text_pos + 1 : 0;
            if (ctx->stream_log && asr_text_pos > 0) {
                char lang[64];
                size_t ll = 0;
                for (int i = 0; i < asr_text_pos; i++) {
                    const char *pc = qwen_tokenizer_decode(tokenizer, raw_tokens[i]);
                    size_t pl = strlen(pc);
                    if (ll + pl >= sizeof(lang)) break;
                    memcpy(lang + ll, pc, pl);
                    ll += pl;
                }
                lang[ll] = 0;
                if (strcmp(lang, logged_lang) != 0) {
                    memcpy(logged_lang, lang, ll + 1);
                    STREAM_EVENT("%s", lang);
                }
            }
        }
        if (text_start < 0) text_start = 0;
        if (text_start > n_raw_tokens) text_start = n_raw_tokens;
        int n_text_tokens = n_raw_tokens - text_start;

        /* "Fixed" frontier for this chunk:
         * - cold-start chunks: emit nothing,
         * - intermediate chunks: keep last `rollback` text tokens unfixed,
         *   but if text is shorter than rollback keep only 1 token unfixed so
         *   streaming still advances,
         * - final chunk: emit everything. */
        int candidate_len = 0;
        if (is_final || is_flush || quiet_reset) {
            candidate_len = n_text_tokens;
            /* The audio of a flush ends at a pause, and the model closes the
             * sentence there; at a hesitation the speaker goes on and the
             * mark is wrong. Hold it back until a second flush confirms the
             * pause; if speech resumes, the next decode decides. A pause long
             * enough for a pause reset needs no confirming. */
            if (is_flush && !is_final && !quiet_reset && !period_held &&
                !(pause_reset_sec > 0 && silent_run_sec >= pause_reset_sec) &&
                n_text_tokens > 0 &&
                stream_piece_ends_sentence(qwen_tokenizer_decode(
                    tokenizer, raw_tokens[text_start + n_text_tokens - 1]))) {
                candidate_len--;
                period_held = 1;
            } else {
                period_held = 0;
            }
        } else if (warmed_up) {
            candidate_len = n_text_tokens - rollback;
            if (candidate_len <= 0 && n_text_tokens > 0) candidate_len = n_text_tokens - 1;
            if (use_agree && candidate_len > agreed_raw - text_start)
                candidate_len = agreed_raw - text_start;
            if (candidate_len < 0) candidate_len = 0;
            period_held = 0;
        }
        /* The frontier never moves behind text already emitted (a chunk that
         * generates fewer than `rollback` new tokens would otherwise do so). */
        if (min_prefix_raw > text_start) {
            int emitted_text = min_prefix_raw - text_start;
            if (emitted_text > n_text_tokens) emitted_text = n_text_tokens;
            if (candidate_len < emitted_text) candidate_len = emitted_text;
        }

        if (stitch_pending && n_emitted_text_tokens == 0) stitch_pending = 0;

        /* Streaming commit: emit token delta against the previous candidate.
         * We do not attempt monotonic byte-level reconciliation here. */
        int *candidate_tokens = raw_tokens + text_start;
        int did_recovery_reset = 0;
        int did_periodic_reset = 0;
        {
            int tail_period = 0;
            int tail_reps = stream_tail_repeat_blocks(candidate_tokens, candidate_len,
                                                      QWEN_STREAM_DEGEN_MAX_PERIOD,
                                                      &tail_period);
            int candidate_advance = candidate_len - n_stable_text_tokens;
            if (!is_final && n_generated >= step_cap && candidate_advance <= 1) {
                stagnant_chunks++;
            } else {
                stagnant_chunks = 0;
            }
            int recovery_reset = 0;
            /* A speaker repeating a word four times is common; a model
             * stuck in a loop also runs into the token limit. */
            if (tail_period > 0 && tail_reps >= QWEN_STREAM_DEGEN_MIN_REPEATS &&
                n_generated >= step_cap) {
                recovery_reset = 1;
            }
            if (stagnant_chunks >= QWEN_STREAM_STALE_CHUNKS) {
                recovery_reset = 1;
            }
            if (dropped_repeat_tokens >= 8) {
                recovery_reset = 1;
            }
            if (force_recovery) {
                recovery_reset = 1;
            }
            if (recovery_reset && qwen_verbose >= 2) {
                fprintf(stderr, "  Recovery cause: tail %dx%d stagnant %d dropped %d%s\n",
                        tail_reps, tail_period, stagnant_chunks, dropped_repeat_tokens,
                        force_recovery ? " loop break" : "");
            }
            char recovery_why[192] = "";
            if (recovery_reset && ctx->stream_log) {
                int w = 0;
                if (force_recovery)
                    w = snprintf(recovery_why, sizeof(recovery_why), "loop while decoding");
                else if (stagnant_chunks >= QWEN_STREAM_STALE_CHUNKS)
                    w = snprintf(recovery_why, sizeof(recovery_why), "no progress");
                else if (dropped_repeat_tokens >= 8)
                    w = snprintf(recovery_why, sizeof(recovery_why), "repeated tokens");
                else
                    w = snprintf(recovery_why, sizeof(recovery_why), "loop of %d", tail_reps);
                if (tail_period > 0 && tail_reps >= 2 && tail_period <= candidate_len) {
                    w += snprintf(recovery_why + w, sizeof(recovery_why) - (size_t)w, ", \"");
                    for (int q = candidate_len - tail_period; q < candidate_len; q++) {
                        const char *pc = qwen_tokenizer_decode(tokenizer, candidate_tokens[q]);
                        size_t room = sizeof(recovery_why) - (size_t)w - 2;
                        size_t pl = strlen(pc);
                        if (pl > room) break;
                        memcpy(recovery_why + w, pc, pl);
                        w += (int)pl;
                    }
                    recovery_why[w] = 0;
                    snprintf(recovery_why + w, sizeof(recovery_why) - (size_t)w, "\"");
                }
            }

            if (recovery_reset && tail_period > 0 &&
                (tail_reps >= QWEN_STREAM_DEGEN_MIN_REPEATS || (force_recovery && tail_reps >= 3))) {
                /* Emit what precedes the loop with one copy of it. Loops
                 * nest ("ああああ。ああああ。"), so cut repeatedly. */
                int reps = tail_reps, period = tail_period;
                for (int pass = 0; pass < 8 && period > 0 && reps >= 3; pass++) {
                    int cut = candidate_len - (reps - 1) * period;
                    candidate_len = cut > 0 ? cut : 0;
                    reps = stream_tail_repeat_blocks(candidate_tokens, candidate_len,
                                                     QWEN_STREAM_DEGEN_MAX_PERIOD, &period);
                }
            }
            /* After a loop recovery, wait for enough new text to align on. */
            if (stitch_pending && stitch_hold && !recovery_reset && !is_final && !is_flush &&
                candidate_len < 8 && chunk_idx - last_recovery_chunk <= recovery_window_chunks) candidate_len = 0;
            {
                if (candidate_len > stable_text_cap) {
                    while (candidate_len > stable_text_cap) stable_text_cap *= 2;
                    int *tmp_stable = (int *)realloc(stable_text_tokens,
                                                     (size_t)stable_text_cap * sizeof(int));
                    if (!tmp_stable) {
                        candidate_len = n_stable_text_tokens;
                    } else {
                        stable_text_tokens = tmp_stable;
                    }
                }

                int lcp = 0;
                while (lcp < n_stable_text_tokens &&
                       lcp < candidate_len &&
                       stable_text_tokens[lcp] == candidate_tokens[lcp]) {
                    lcp++;
                }
                for (int i = lcp; i < candidate_len; i++) {
                    stable_text_tokens[i] = candidate_tokens[i];
                }

                /* Nothing has been emitted since the restart. */
                int emit_start = stitch_pending ? 0 : lcp;
                if (emit_start < candidate_len && n_emitted_text_tokens > 0) {
                    int max_overlap = candidate_len - emit_start;
                    if (max_overlap > n_emitted_text_tokens) max_overlap = n_emitted_text_tokens;
                    if (max_overlap > QWEN_STREAM_OVERLAP_MAX_TOKENS)
                        max_overlap = QWEN_STREAM_OVERLAP_MAX_TOKENS;
                    for (int k = max_overlap; k >= QWEN_STREAM_OVERLAP_MIN_TOKENS; k--) {
                        if (memcmp(emitted_text_tokens + (n_emitted_text_tokens - k),
                                   candidate_tokens + emit_start,
                                   (size_t)k * sizeof(int)) == 0) {
                            emit_start += k;
                            break;
                        }
                    }
                }

                if (stitch_pending && candidate_len > 0) {
                    /* Align a suffix of the emitted text with a prefix of
                     * the new text, so rewording ("3000個" / "三千個") and a
                     * looped copy at the end still match. */
                    enum { ST_T = QWEN_STREAM_STITCH_TAIL, ST_C = 160 };
                    static int st[ST_T + 1][ST_C + 1];
                    int tn = n_emitted_text_tokens < ST_T ? n_emitted_text_tokens : ST_T;
                    const int *tail = emitted_text_tokens + (n_emitted_text_tokens - tn);
                    int cm = candidate_len < ST_C ? candidate_len : ST_C;
                    for (int x = 0; x <= tn; x++) st[x][0] = 0;       /* free start in the tail */
                    for (int y = 1; y <= cm; y++) st[0][y] = -y;
                    for (int x = 1; x <= tn; x++) {
                        for (int y = 1; y <= cm; y++) {
                            int d = st[x - 1][y - 1] +
                                    (tail[x - 1] == candidate_tokens[y - 1] ? 2 : -1);
                            int u = st[x - 1][y] - 1;
                            int l = st[x][y - 1] - 1;
                            int b = d > u ? d : u;
                            st[x][y] = b > l ? b : l;
                        }
                    }
                    int best = 0, best_y = 0;                /* old part ends at best_y */
                    for (int y = 1; y <= cm; y++) {
                        if (st[tn][y] > best) { best = st[tn][y]; best_y = y; }
                    }
                    /* New text lying wholly inside the emitted text has
                     * nothing new yet; decide on a later step. */
                    int inside = 0;
                    for (int x = 1; x <= tn; x++) if (st[x][cm] > inside) inside = st[x][cm];
                    stitch_tries++;
                    int undecided = (inside > best && inside >= QWEN_STREAM_STITCH_MIN_SCORE &&
                                     !is_final && !is_flush && !recovery_reset &&
                                     stitch_tries <= recovery_window_chunks);
                    if (inside > best && inside >= QWEN_STREAM_STITCH_MIN_SCORE) {
                        best_y = cm;
                        best = inside;
                    }
                    if (best >= QWEN_STREAM_STITCH_MIN_SCORE && best_y > emit_start) emit_start = best_y;
                    if (qwen_verbose >= 2)
                        fprintf(stderr, "  Stitch: score %d, skip %d of %d%s\n",
                                best, emit_start, candidate_len, undecided ? " (undecided)" : "");
                    if (undecided) emit_start = candidate_len;
                    else stitch_pending = 0;
                }

                upd_commit.len = 0;
                if (ctx->stream_update_cb && upd_pending_len > 0) {
                    stream_str_append(&upd_commit, upd_pending, upd_pending_len);
                }
                if (seam_space && emit_start < candidate_len) {
                    const char *first = qwen_tokenizer_decode(tokenizer, stable_text_tokens[emit_start]);
                    if (result_len > 0 &&
                        stream_seam_needs_space((unsigned char)result[result_len - 1], first)) {
                        if (ctx->token_cb) ctx->token_cb(" ", ctx->token_cb_userdata);
                        if (ctx->stream_update_cb) stream_str_append(&upd_commit, " ", 1);
                        char *grown = result;
                        if (result_len + 2 > result_cap) {
                            grown = (char *)realloc(result, result_cap * 2);
                            if (grown) { result = grown; result_cap *= 2; }
                        }
                        if (grown) {
                            result[result_len++] = ' ';
                            result[result_len] = '\0';
                        }
                    }
                    seam_space = 0;
                }
                for (int i = emit_start; i < candidate_len; i++) {
                    int tok = stable_text_tokens[i];
                    const char *piece = qwen_tokenizer_decode(tokenizer, tok);
                    if (ctx->token_cb) ctx->token_cb(piece, ctx->token_cb_userdata);

                    size_t plen = strlen(piece);
                    if (ctx->stream_update_cb) stream_str_append(&upd_commit, piece, plen);
                    if (result_len + plen + 1 > result_cap) {
                        size_t cap = result_cap;
                        while (result_len + plen + 1 > cap) cap *= 2;
                        char *grown = (char *)realloc(result, cap);
                        if (grown) { result = grown; result_cap = cap; }
                    }
                    if (result_len + plen + 1 <= result_cap) {
                        memcpy(result + result_len, piece, plen);
                        result_len += plen;
                        result[result_len] = '\0';
                    }
                    ctx->perf_text_tokens++;

                    if (n_emitted_text_tokens == emitted_text_cap) {
                        int new_cap = emitted_text_cap * 2;
                        int *tmp_emit = (int *)realloc(emitted_text_tokens,
                                                       (size_t)new_cap * sizeof(int));
                        if (tmp_emit) {
                            emitted_text_tokens = tmp_emit;
                            emitted_text_cap = new_cap;
                        }
                    }
                    if (n_emitted_text_tokens < emitted_text_cap) {
                        emitted_text_tokens[n_emitted_text_tokens++] = tok;
                    }
                }

                n_stable_text_tokens = candidate_len;
                held_text_tokens = n_text_tokens - candidate_len;
                if (held_text_tokens == 0 && !stitch_pending) emitted_cursor = decoded_cursor;
                if (ctx->past_text_conditioning && candidate_len > 0 &&
                    text_start + candidate_len > min_prefix_raw) {
                    min_prefix_raw = text_start + candidate_len;
                }

                if (ctx->stream_update_cb) {
                    /* Commit only whole characters; a split one waits in
                     * upd_pending and leads the tail meanwhile. */
                    size_t commit_len = is_final ? upd_commit.len
                        : stream_utf8_complete_len(upd_commit.buf ? upd_commit.buf : "",
                                                   upd_commit.len);
                    upd_pending_len = upd_commit.len - commit_len;
                    if (upd_pending_len > 0) {
                        memcpy(upd_pending, upd_commit.buf + commit_len, upd_pending_len);
                    }
                    upd_tail.len = 0;
                    if (upd_pending_len > 0) {
                        stream_str_append(&upd_tail, upd_pending, upd_pending_len);
                    }
                    for (int i = candidate_len; i < n_text_tokens; i++) {
                        const char *piece = qwen_tokenizer_decode(tokenizer, candidate_tokens[i]);
                        stream_str_append(&upd_tail, piece, strlen(piece));
                    }
                    upd_tail.len = stream_utf8_complete_len(upd_tail.buf ? upd_tail.buf : "",
                                                            upd_tail.len);
                    int tail_changed = upd_tail.len != upd_last_tail.len ||
                        (upd_tail.len > 0 &&
                         memcmp(upd_tail.buf, upd_last_tail.buf, upd_tail.len) != 0);
                    if (commit_len > 0 || tail_changed) {
                        char *commit_str = (char *)malloc(commit_len + 1);
                        char *tail_str = (char *)malloc(upd_tail.len + 1);
                        if (commit_str && tail_str) {
                            if (commit_len > 0) memcpy(commit_str, upd_commit.buf, commit_len);
                            commit_str[commit_len] = '\0';
                            if (upd_tail.len > 0) memcpy(tail_str, upd_tail.buf, upd_tail.len);
                            tail_str[upd_tail.len] = '\0';
                            ctx->stream_update_cb(commit_str, tail_str,
                                                  ctx->stream_update_cb_userdata);
                        }
                        free(commit_str);
                        free(tail_str);
                        upd_last_tail.len = 0;
                        if (upd_tail.len > 0) {
                            stream_str_append(&upd_last_tail, upd_tail.buf, upd_tail.len);
                        }
                    }
                }

                int periodic_reset =
                    (!is_final && !recovery_reset && !period_held &&
                     ctx->past_text_conditioning &&
                     chunk_idx >= unfixed_chunks &&
                     chunk_idx - last_trim_chunk >= reset_interval_chunks);
                /* Carried text without punctuation keeps the model writing
                 * none (1.7B drifts into it); then carry no text. */
                int periodic_no_text = 0;
                if (periodic_reset && n_emitted_text_tokens >= QWEN_STREAM_RESET_CARRY_TOKENS) {
                    int has_punct = 0;
                    for (int q = n_emitted_text_tokens - QWEN_STREAM_RESET_CARRY_TOKENS;
                         q < n_emitted_text_tokens && !has_punct; q++) {
                        const char *pc = qwen_tokenizer_decode(tokenizer, emitted_text_tokens[q]);
                        if (strstr(pc, "\xe3\x80\x82") || strstr(pc, "\xe3\x80\x81") ||   /* 。、 */
                            strstr(pc, "\xef\xbc\x8c") || strstr(pc, "\xef\xbc\x9f") ||   /* ，？ */
                            strstr(pc, "\xef\xbc\x81") || strpbrk(pc, ".,?!")) has_punct = 1; /* ！ */
                    }
                    periodic_no_text = !has_punct;
                }
                if (periodic_reset && periodic_no_text) {
                    STREAM_RESET_TEXT(text_start);
                    STREAM_RESET_AUDIO(QWEN_STREAM_RESET_KEEP_ENC_WINDOWS, 0);
                    STREAM_REALIGN(0);
                    STREAM_RESET_DONE();
                    did_periodic_reset = 1;
                    STREAM_EVENT("periodic reset, no text kept (none of it punctuated)");
                } else if (periodic_reset) {
                    if (stream_reanchor_text_state(ctx,
                                                   emitted_text_tokens,
                                                   n_emitted_text_tokens,
                                                   QWEN_STREAM_RESET_CARRY_TOKENS,
                                                   &raw_tokens, &raw_tokens_cap, &n_raw_tokens,
                                                   &stable_text_tokens, &stable_text_cap,
                                                   &n_stable_text_tokens) != 0) {
                        n_raw_tokens = 0;
                        n_stable_text_tokens = 0;
                    }
                    /* Carried tokens are already shown; left open they get
                     * rewritten and both versions show. */
                    min_prefix_raw = n_raw_tokens;
                    agreed_raw = n_raw_tokens;
                    held_text_tokens = 0;
                    prev_prefill_len = 0;
                    STREAM_RESET_AUDIO(QWEN_STREAM_RESET_KEEP_ENC_WINDOWS, 0);
                    STREAM_RESET_DONE();
                    did_periodic_reset = 1;
                    STREAM_EVENT("periodic reset");
                }
                /* New words in a chunk with no speech frame: speech too
                 * quiet for the frame test (low-SNR radio, say). Not a pause. */
                if (chunk_no_speech &&
                    n_emitted_text_tokens + held_text_tokens >= known_text_before + 2) {
                    trail_sil_sec = 0;
                    silent_run_sec = 0;
                    pause_reset_done = 0;
                }
                /* A decoded chunk can already end in a long enough pause
                 * (with 4 s chunks a single one can). */
                if ((is_flush || held_text_tokens == 0) && !did_periodic_reset && !recovery_reset && !quiet_reset &&
                    !pause_reset_done && pause_reset_sec > 0 &&
                    silent_run_sec >= pause_reset_sec) {
                    STREAM_PAUSE_RESET();
                }
                if (quiet_reset && !did_periodic_reset && !recovery_reset) {
                    /* Keep only the language header the model produced, so
                     * the restart does not re-detect (and switch) language. */
                    STREAM_RESET_TEXT(text_start);
                    STREAM_RESET_AUDIO(-1, full_end);
                    quiet_chunks = 0;
                    STREAM_REALIGN(0);
                    STREAM_RESET_DONE();
                    if (qwen_verbose >= 2) fprintf(stderr, "  Stall reset applied\n");
                    STREAM_EVENT("stall reset: %d steps without new text",
                                 QWEN_STREAM_QUIET_RESET_CHUNKS);
                }
            }
            if (recovery_reset) {
                /* Carry no text: the emitted tail is the loop itself. */
                STREAM_RESET_TEXT(text_start);
                if (chunk_idx - last_recovery_chunk <= recovery_window_chunks) {
                    /* Second loop in a row on the same audio: drop it. */
                    STREAM_RESET_AUDIO(-1, full_end);
                    stitch_pending = 0;
                    if (qwen_verbose >= 2) fprintf(stderr, "  Recovery mode: hard\n");
                    STREAM_EVENT("hard recovery (audio dropped): %s", recovery_why);
                } else {
                    /* Keep the last window so the words around the loop
                     * are decoded again. */
                    STREAM_RESET_AUDIO(QWEN_STREAM_RESET_KEEP_ENC_WINDOWS, 0);
                    STREAM_REALIGN(1);
                    if (qwen_verbose >= 2) fprintf(stderr, "  Recovery mode: soft\n");
                    STREAM_EVENT("recovery: %s", recovery_why);
                }
                last_recovery_chunk = chunk_idx;
                STREAM_RESET_DONE();
                stagnant_chunks = 0;
                quiet_chunks = 0;
                did_recovery_reset = 1;
                if (ctx->stream_update_cb && upd_last_tail.len > 0) {
                    /* The held tokens were discarded with the reset. */
                    ctx->stream_update_cb("", "", ctx->stream_update_cb_userdata);
                    upd_last_tail.len = 0;
                }
                if (qwen_monitor) {
                    fprintf(stderr, "!");
                    fflush(stderr);
                }
            }
        }

        if (qwen_verbose >= 2) {
            if (prefix_offset > 0)
                fprintf(stderr, "  Prefix window: %d/%d tokens (offset %d)\n",
                        n_prefix_tokens, n_prefix_tokens_full, prefix_offset);
            if (did_recovery_reset) {
                fprintf(stderr, "  Recovery reset applied\n");
            } else if (did_periodic_reset) {
                fprintf(stderr, "  Periodic reset applied\n");
            }
            fprintf(stderr, "  Commit: candidate=%d tokens, emitted_total=%d\n",
                    candidate_len, n_stable_text_tokens);
        }

        if (live && use_enc_cache) {
            /* Keep only the current partial tail [full_end, audio_n_samples). */
            int64_t keep_from = full_end;
            if (keep_from > local_base_sample) {
                int64_t drop64 = keep_from - local_base_sample;
                if (drop64 > local_n_samples) drop64 = local_n_samples;
                if (drop64 > 0) {
                    int64_t remain = local_n_samples - drop64;
                    if (remain > 0) {
                        memmove(local_samples, local_samples + (size_t)drop64,
                                (size_t)remain * sizeof(float));
                    }
                    local_n_samples = remain;
                    local_base_sample += drop64;
                    audio_samples = local_samples;
                    audio_n_samples = local_base_sample + local_n_samples;
                }
            }
        }

        ctx->perf_total_ms += get_time_ms() - chunk_t0;
        chunk_idx++;
    }

    free(tmp_embed);
    for (int i = enc_cache_start; i < n_enc_cache; i++) {
        free(enc_cache[i].enc_output);
    }
    free(enc_cache);
    if (qwen_verbose >= 2 && prefill_total_tokens > 0) {
        double reuse_pct = 100.0 * (double)prefill_reused_tokens / (double)prefill_total_tokens;
        fprintf(stderr, "  Prefill reuse: %d/%d tokens (%.1f%%)\n",
                prefill_reused_tokens, prefill_total_tokens, reuse_pct);
    }
    free(prev_prefill_embeds);
    free(raw_tokens);
    free(stable_text_tokens);
    free(emitted_text_tokens);
    free(upd_commit.buf);
    free(ep_hist);
    free(ep_sorted);
    free(upd_tail.buf);
    free(upd_last_tail.buf);
    qwen_tokenizer_free(tokenizer);
    free(compacted_samples);
    free(local_samples);

    /* Trim whitespace */
    size_t rlen = strlen(result);
    while (rlen > 0 && isspace((unsigned char)result[rlen - 1])) result[--rlen] = '\0';
    char *start = result;
    while (*start && isspace((unsigned char)*start)) start++;
    if (start != result) memmove(result, start, strlen(start) + 1);

    return result;
}

char *qwen_transcribe_stream(qwen_ctx_t *ctx, const float *samples, int n_samples) {
    return stream_impl(ctx, samples, n_samples, NULL);
}

char *qwen_transcribe_stream_live(qwen_ctx_t *ctx, qwen_live_audio_t *live) {
    return stream_impl(ctx, NULL, 0, live);
}

char *qwen_transcribe(qwen_ctx_t *ctx, const char *wav_path) {
    int n_samples = 0;
    float *samples = qwen_load_wav(wav_path, &n_samples);
    if (!samples) {
        fprintf(stderr, "qwen_transcribe: cannot load %s\n", wav_path);
        return NULL;
    }
    char *text = qwen_transcribe_audio(ctx, samples, n_samples);
    free(samples);
    return text;
}

char *qwen_transcribe_stdin(qwen_ctx_t *ctx) {
    int n_samples = 0;
    float *samples = qwen_read_pcm_stdin(&n_samples);
    if (!samples) return NULL;
    char *text = qwen_transcribe_audio(ctx, samples, n_samples);
    free(samples);
    return text;
}
