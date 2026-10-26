/*
 * qwen_asr_ggml.c - encoder and decoder on a ggml GPU (--device)
 *
 * The same computation as qwen_encoder_forward and qwen_decoder_forward,
 * built as a ggml graph per call. Weights are converted to f16 once at load;
 * the decoder KV cache is f16 on the device and indexed like the CPU one, so
 * callers keep moving ctx->kv_cache_len to reuse or drop a prefix.
 */

#ifdef USE_GGML

#include "qwen_asr_ggml.h"
#include "qwen_asr_kernels.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define GRAPH_MAX 8192

typedef struct {
    struct ggml_tensor *wq, *wk, *wv, *wo;
    struct ggml_tensor *q_norm, *k_norm, *in_norm, *post_norm;
    struct ggml_tensor *gate, *up, *down;
} gdec_layer_t;

typedef struct {
    ggml_backend_t dev, cpu;
    ggml_backend_sched_t sched;
    struct ggml_context *probe_ctx;
    struct ggml_cgraph *probe;      /* empty graph, see compute() */

    struct ggml_context *wctx;
    ggml_backend_buffer_t wbuf;
    struct ggml_tensor *tok_emb;    /* [dim, vocab] f16, tied lm_head */
    struct ggml_tensor *norm;
    gdec_layer_t layers[QWEN_MAX_DEC_LAYERS];

    struct ggml_context *kvctx;
    ggml_backend_buffer_t kvbuf;
    struct ggml_tensor *k[QWEN_MAX_DEC_LAYERS], *v[QWEN_MAX_DEC_LAYERS];
    int kv_max;

    void *meta;                     /* graph-building arena */
    size_t meta_size;
    ggml_fp16_t *mask;
    size_t mask_cap;
    float *logits;                  /* [rows][vocab] */
    size_t logits_cap;

    /* Encoder, on the same device unless QWEN_GPU_ENC=0 */
    struct ggml_context *ectx;
    ggml_backend_buffer_t ebuf;
    struct ggml_tensor *c1w, *c1b, *c2w, *c2b, *c3w, *c3b, *conv_out;
    struct ggml_tensor *ln_post_w, *ln_post_b, *p1w, *p1b, *p2w, *p2b;
    struct {
        struct ggml_tensor *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo;
        struct ggml_tensor *ln1_w, *ln1_b, *fc1, *b1, *fc2, *b2, *ln2_w, *ln2_b;
    } enc[QWEN_MAX_ENC_LAYERS];
} gdec_t;

static int upload_f32_as_f16(struct ggml_tensor *t, const float *src) {
    size_t n = (size_t)ggml_nelements(t);
    const size_t chunk = 1 << 20;
    ggml_fp16_t *h = (ggml_fp16_t *)malloc(chunk * sizeof(ggml_fp16_t));
    if (!h) return -1;
    for (size_t off = 0; off < n; off += chunk) {
        size_t m = n - off < chunk ? n - off : chunk;
        ggml_fp32_to_fp16_row(src + off, h, (int64_t)m);
        ggml_backend_tensor_set(t, h, off * sizeof(ggml_fp16_t), m * sizeof(ggml_fp16_t));
    }
    free(h);
    return 0;
}

static void upload_f32(struct ggml_tensor *t, const float *src) {
    ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
}

static int upload_bf16(struct ggml_tensor *t, const uint16_t *src) {
    size_t n = (size_t)ggml_nelements(t);
    const size_t chunk = 1 << 20;
    float *f = (float *)malloc(chunk * sizeof(float));
    ggml_fp16_t *h = (ggml_fp16_t *)malloc(chunk * sizeof(ggml_fp16_t));
    if (!f || !h) { free(f); free(h); return -1; }
    for (size_t off = 0; off < n; off += chunk) {
        size_t m = n - off < chunk ? n - off : chunk;
        ggml_bf16_to_fp32_row((const ggml_bf16_t *)src + off, f, (int64_t)m);
        ggml_fp32_to_fp16_row(f, h, (int64_t)m);
        ggml_backend_tensor_set(t, h, off * sizeof(ggml_fp16_t), m * sizeof(ggml_fp16_t));
    }
    free(f);
    free(h);
    return 0;
}

static int kv_alloc(gdec_t *g, const qwen_config_t *cfg, int kv_max) {
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    struct ggml_init_params p = { (size_t)2 * cfg->dec_layers * ggml_tensor_overhead(), NULL, true };
    struct ggml_context *c = ggml_init(p);
    if (!c) return -1;
    struct ggml_tensor *k[QWEN_MAX_DEC_LAYERS], *v[QWEN_MAX_DEC_LAYERS];
    for (int l = 0; l < cfg->dec_layers; l++) {
        k[l] = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_dim, kv_max);
        v[l] = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_dim, kv_max);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, g->dev);
    if (!buf) { ggml_free(c); return -1; }

    /* Keep what the old cache held (rare: the context outgrew it). */
    if (g->kvbuf) {
        size_t row = (size_t)kv_dim * sizeof(ggml_fp16_t);
        size_t bytes = row * (size_t)g->kv_max;
        void *tmp = malloc(bytes);
        if (!tmp) { ggml_backend_buffer_free(buf); ggml_free(c); return -1; }
        for (int l = 0; l < cfg->dec_layers; l++) {
            ggml_backend_tensor_get(g->k[l], tmp, 0, bytes);
            ggml_backend_tensor_set(k[l], tmp, 0, bytes);
            ggml_backend_tensor_get(g->v[l], tmp, 0, bytes);
            ggml_backend_tensor_set(v[l], tmp, 0, bytes);
        }
        free(tmp);
        ggml_backend_buffer_free(g->kvbuf);
        ggml_free(g->kvctx);
    }
    g->kvctx = c;
    g->kvbuf = buf;
    memcpy(g->k, k, sizeof(k));
    memcpy(g->v, v, sizeof(v));
    g->kv_max = kv_max;
    return 0;
}

/* ggml logs to stderr by default; keep it to errors unless --debug. */
static void qwen_ggml_log(enum ggml_log_level level, const char *text, void *user) {
    static enum ggml_log_level last = GGML_LOG_LEVEL_NONE;
    (void)user;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (qwen_verbose >= 2 || (qwen_verbose >= 1 && last == GGML_LOG_LEVEL_ERROR))
        fputs(text, stderr);
}

static void ggml_setup(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    ggml_log_set(qwen_ggml_log, NULL);
#ifdef GGML_BACKEND_DL
    /* Backend modules next to the binary; without a driver only the CPU loads */
    ggml_backend_load_all();
#endif
}

/* --device gpu: the first discrete GPU, else the first integrated one. ggml's
 * CPU backend is never offered; this program's CPU path is faster. */
static ggml_backend_dev_t default_device(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    return dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
}

/* ggml's CPU backend (ops the GPU lacks) defaults to 4 threads. */
static void set_threads(ggml_backend_t b) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(b));
    ggml_backend_set_n_threads_t fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn) fn(b, qwen_get_threads());
}

static void print_json_string(const char *s) {
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') putchar('\\');
        if ((unsigned char)*s >= 0x20) putchar(*s);
    }
    putchar('"');
}

int qwen_ggml_list_devices(int json) {
    ggml_setup();
    ggml_backend_dev_t def = default_device();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        size_t free_mem = 0, total = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total);
        int igpu = type == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (json) {
            printf(",{\"name\":");
            print_json_string(ggml_backend_dev_name(dev));
            printf(",\"description\":");
            print_json_string(ggml_backend_dev_description(dev));
            printf(",\"type\":\"%s\",\"free_mib\":%zu,\"total_mib\":%zu,\"default\":false,\"gpu_default\":%s}",
                   igpu ? "igpu" : "gpu", free_mem >> 20, total >> 20, dev == def ? "true" : "false");
        } else {
            printf("%s: %s (%s, %zu MiB free of %zu MiB)%s\n", ggml_backend_dev_name(dev),
                   ggml_backend_dev_description(dev), igpu ? "integrated gpu" : "gpu",
                   free_mem >> 20, total >> 20, dev == def ? " [gpu]" : "");
        }
    }
    return 0;
}

int qwen_ggml_decoder_init(qwen_ctx_t *ctx) {
    const qwen_config_t *cfg = &ctx->config;
    qwen_decoder_t *dec = &ctx->decoder;

    ggml_setup();
    ggml_backend_dev_t dev;
    if (strcasecmp(qwen_device, "gpu") == 0) {
        dev = default_device();
        if (!dev) return -1;
    } else {
        dev = ggml_backend_dev_by_name(qwen_device);
        if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            fprintf(stderr, "qwen_load: no device named %s (see --list-devices)\n", qwen_device);
            return -2;
        }
    }

    gdec_t *g = (gdec_t *)calloc(1, sizeof(*g));
    if (!g) return -1;
    g->dev = ggml_backend_dev_init(dev, NULL);
    g->cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!g->dev || !g->cpu) {
        fprintf(stderr, "ggml: cannot initialize %s\n", ggml_backend_dev_name(dev));
        if (g->dev) ggml_backend_free(g->dev);
        if (g->cpu) ggml_backend_free(g->cpu);
        free(g);
        return -1;
    }
    set_threads(g->dev);
    set_threads(g->cpu);
    if (qwen_verbose >= 1)
        fprintf(stderr, "GPU: %s\n", ggml_backend_name(g->dev));

    int dim = cfg->dec_hidden;
    int q_dim = cfg->dec_heads * cfg->dec_head_dim;
    int kv_dim = cfg->dec_kv_heads * cfg->dec_head_dim;
    int inter = cfg->dec_intermediate;
    int hd = cfg->dec_head_dim;

    struct ggml_init_params p = { (size_t)(16 * cfg->dec_layers + 4) * ggml_tensor_overhead(), NULL, true };
    g->wctx = ggml_init(p);
    struct ggml_context *c = g->wctx;
    g->tok_emb = ggml_new_tensor_2d(c, GGML_TYPE_F16, dim, cfg->vocab_size);
    g->norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim);
    for (int i = 0; i < cfg->dec_layers; i++) {
        gdec_layer_t *L = &g->layers[i];
        L->wq = ggml_new_tensor_2d(c, GGML_TYPE_F16, dim, q_dim);
        L->wk = ggml_new_tensor_2d(c, GGML_TYPE_F16, dim, kv_dim);
        L->wv = ggml_new_tensor_2d(c, GGML_TYPE_F16, dim, kv_dim);
        L->wo = ggml_new_tensor_2d(c, GGML_TYPE_F16, q_dim, dim);
        L->gate = ggml_new_tensor_2d(c, GGML_TYPE_F16, dim, inter);
        L->up = ggml_new_tensor_2d(c, GGML_TYPE_F16, dim, inter);
        L->down = ggml_new_tensor_2d(c, GGML_TYPE_F16, inter, dim);
        L->q_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
        L->k_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, hd);
        L->in_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim);
        L->post_norm = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim);
    }
    g->wbuf = ggml_backend_alloc_ctx_tensors(c, g->dev);
    if (!g->wbuf) { fprintf(stderr, "ggml: weight allocation failed\n"); free(g); return -1; }

    int bad = upload_bf16(g->tok_emb, dec->tok_embeddings_bf16);
    ggml_backend_tensor_set(g->norm, dec->norm, 0, (size_t)dim * sizeof(float));
    for (int i = 0; i < cfg->dec_layers && !bad; i++) {
        qwen_dec_layer_t *l = &dec->layers[i];
        gdec_layer_t *L = &g->layers[i];
        bad |= upload_bf16(L->wq, l->wq_weight_bf16) | upload_bf16(L->wk, l->wk_weight_bf16) |
               upload_bf16(L->wv, l->wv_weight_bf16) | upload_bf16(L->wo, l->wo_weight_bf16) |
               upload_bf16(L->gate, l->gate_weight_bf16) | upload_bf16(L->up, l->up_weight_bf16) |
               upload_bf16(L->down, l->down_weight_bf16);
        ggml_backend_tensor_set(L->q_norm, l->q_norm_weight, 0, (size_t)hd * sizeof(float));
        ggml_backend_tensor_set(L->k_norm, l->k_norm_weight, 0, (size_t)hd * sizeof(float));
        ggml_backend_tensor_set(L->in_norm, l->input_norm, 0, (size_t)dim * sizeof(float));
        ggml_backend_tensor_set(L->post_norm, l->post_attn_norm, 0, (size_t)dim * sizeof(float));
    }
    if (bad || kv_alloc(g, cfg, 2048) != 0) { fprintf(stderr, "ggml: upload failed\n"); free(g); return -1; }

    ggml_backend_t backends[2] = { g->dev, g->cpu };
    g->sched = ggml_backend_sched_new(backends, NULL, 2, GRAPH_MAX, false, true);
    struct ggml_init_params pp = { ggml_graph_overhead(), NULL, true };
    g->probe_ctx = ggml_init(pp);
    g->probe = ggml_new_graph(g->probe_ctx);
    g->meta_size = ggml_tensor_overhead() * GRAPH_MAX + ggml_graph_overhead_custom(GRAPH_MAX, false);
    g->meta = malloc(g->meta_size);
    if (!g->sched || !g->meta) { free(g); return -1; }

    ctx->ggml_dec = g;
    return 0;
}

/* Metal runs a graph asynchronously and reports a failed command buffer (out
 * of memory, say) only by putting the backend into an error state, which the
 * next graph sees. An empty graph after each one finds out at once. */
static int compute(gdec_t *g, struct ggml_cgraph *gf) {
    return ggml_backend_sched_graph_compute(g->sched, gf) == GGML_STATUS_SUCCESS &&
           ggml_backend_graph_compute(g->dev, g->probe) == GGML_STATUS_SUCCESS;
}

void qwen_ggml_lost(qwen_ctx_t *ctx) {
    (void)ctx;
    fprintf(stderr, "GPU computation failed (out of GPU memory?); run with --device cpu\n");
    /* _Exit: ggml's device destructors assert on the buffers still held */
    fflush(NULL);
    _Exit(1);
}

void qwen_ggml_decoder_free(qwen_ctx_t *ctx) {
    gdec_t *g = (gdec_t *)ctx->ggml_dec;
    if (!g) return;
    ggml_backend_sched_free(g->sched);
    if (g->probe_ctx) ggml_free(g->probe_ctx);
    ggml_backend_buffer_free(g->kvbuf);
    ggml_free(g->kvctx);
    ggml_backend_buffer_free(g->wbuf);
    ggml_free(g->wctx);
    if (g->ebuf) ggml_backend_buffer_free(g->ebuf);
    if (g->ectx) ggml_free(g->ectx);
    ggml_backend_free(g->cpu);
    ggml_backend_free(g->dev);
    free(g->meta);
    free(g->mask);
    free(g->logits);
    free(g);
    ctx->ggml_dec = NULL;
}

/* --debug: once per graph kind, where its nodes ran. */
static void report_sched(gdec_t *g, struct ggml_cgraph *gf, const char *what) {
    static int seen[3];
    int k = what[0] == 'e' ? 0 : what[0] == 'p' ? 1 : 2;
    if (qwen_verbose < 2 || seen[k]++) return;
    int n_dev = 0, n_cpu = 0, n_other = 0;
    fprintf(stderr, "ggml %s: %d nodes, %d splits, %d copies\n", what, ggml_graph_n_nodes(gf),
            ggml_backend_sched_get_n_splits(g->sched), ggml_backend_sched_get_n_copies(g->sched));
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        struct ggml_tensor *t = ggml_graph_node(gf, i);
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(g->sched, t);
        if (b == g->dev) n_dev++;
        else if (b == g->cpu) { n_cpu++; fprintf(stderr, "  on CPU: %s (%s)\n", ggml_op_desc(t), t->name); }
        else n_other++;
    }
    fprintf(stderr, "  %s: %d nodes, CPU: %d, unassigned: %d\n", ggml_backend_name(g->dev), n_dev, n_cpu, n_other);
}

static struct ggml_tensor *norm_mul(struct ggml_context *c, struct ggml_tensor *x,
                                    struct ggml_tensor *w, float eps) {
    return ggml_mul(c, ggml_rms_norm(c, x, eps), w);
}

int qwen_ggml_decode(qwen_ctx_t *ctx, const float *embeds, int n, int want_logits) {
    gdec_t *g = (gdec_t *)ctx->ggml_dec;
    const qwen_config_t *cfg = &ctx->config;
    int dim = cfg->dec_hidden;
    int n_heads = cfg->dec_heads, n_kv = cfg->dec_kv_heads, hd = cfg->dec_head_dim;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    float eps = cfg->dec_rms_norm_eps;
    int p0 = ctx->kv_cache_len;
    int kv_len = p0 + n;

    if (kv_len > g->kv_max) {
        int m = g->kv_max;
        while (m < kv_len) m *= 2;
        if (kv_alloc(g, cfg, m) != 0) return -1;
    }

    struct ggml_init_params p = { g->meta_size, g->meta, true };
    struct ggml_context *c = ggml_init(p);
    struct ggml_cgraph *gf = ggml_new_graph_custom(c, GRAPH_MAX, false);

    struct ggml_tensor *inp = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, n);
    struct ggml_tensor *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
    struct ggml_tensor *mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_len, n);
    ggml_set_input(inp);
    ggml_set_input(pos);
    ggml_set_input(mask);

    float scale = 1.0f / sqrtf((float)hd);
    struct ggml_tensor *x = inp;
    for (int layer = 0; layer < cfg->dec_layers; layer++) {
        gdec_layer_t *L = &g->layers[layer];
        int last = layer == cfg->dec_layers - 1;

        struct ggml_tensor *h = norm_mul(c, x, L->in_norm, eps);
        struct ggml_tensor *q = ggml_reshape_3d(c, ggml_mul_mat(c, L->wq, h), hd, n_heads, n);
        struct ggml_tensor *k = ggml_reshape_3d(c, ggml_mul_mat(c, L->wk, h), hd, n_kv, n);
        struct ggml_tensor *v = ggml_mul_mat(c, L->wv, h);
        q = norm_mul(c, q, L->q_norm, eps);
        k = norm_mul(c, k, L->k_norm, eps);
        q = ggml_rope_ext(c, q, pos, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cfg->dec_rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(c, k, pos, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, cfg->dec_rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        /* Write rows p0..p0+n-1; expanded before the attention that reads them. */
        size_t row = g->k[layer]->nb[1];
        ggml_build_forward_expand(gf, ggml_cpy(c, ggml_reshape_2d(c, k, kv_dim, n),
                                              ggml_view_2d(c, g->k[layer], kv_dim, n, row, (size_t)p0 * row)));
        ggml_build_forward_expand(gf, ggml_cpy(c, v,
                                              ggml_view_2d(c, g->v[layer], kv_dim, n, row, (size_t)p0 * row)));
        if (last && !want_logits) break;   /* only the KV of the last layer is needed */

        struct ggml_tensor *kc = ggml_permute(c, ggml_view_3d(c, g->k[layer], hd, n_kv, kv_len,
                                                             (size_t)hd * sizeof(ggml_fp16_t), row, 0),
                                              0, 2, 1, 3);
        struct ggml_tensor *vc = ggml_permute(c, ggml_view_3d(c, g->v[layer], hd, n_kv, kv_len,
                                                             (size_t)hd * sizeof(ggml_fp16_t), row, 0),
                                              0, 2, 1, 3);
        struct ggml_tensor *a = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), kc, vc, mask,
                                                    scale, 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_reshape_2d(c, a, q_dim, n);
        x = ggml_add(c, x, ggml_mul_mat(c, L->wo, a));

        h = norm_mul(c, x, L->post_norm, eps);
        struct ggml_tensor *m = ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, L->gate, h)),
                                         ggml_mul_mat(c, L->up, h));
        x = ggml_add(c, x, ggml_mul_mat(c, L->down, m));
    }
    struct ggml_tensor *logits = NULL;
    if (want_logits) {
        logits = ggml_mul_mat(c, g->tok_emb, norm_mul(c, x, g->norm, eps));
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    }

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, gf)) { ggml_free(c); return -1; }

    ggml_backend_tensor_set(inp, embeds, 0, (size_t)n * dim * sizeof(float));
    int32_t *pv = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    for (int i = 0; i < n; i++) pv[i] = p0 + i;
    ggml_backend_tensor_set(pos, pv, 0, (size_t)n * sizeof(int32_t));
    free(pv);
    size_t mn = (size_t)n * kv_len;
    if (mn > g->mask_cap) {
        free(g->mask);
        g->mask = (ggml_fp16_t *)malloc(mn * sizeof(ggml_fp16_t));
        g->mask_cap = g->mask ? mn : 0;
        if (!g->mask) { ggml_free(c); return -1; }
    }
    const uint16_t neg_inf = 0xFC00, zero = 0;   /* f16 -inf, 0 */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < kv_len; j++)
            memcpy(&g->mask[(size_t)i * kv_len + j], j <= p0 + i ? &zero : &neg_inf, 2);
    ggml_backend_tensor_set(mask, g->mask, 0, mn * sizeof(ggml_fp16_t));

    if (!compute(g, gf)) { ggml_free(c); qwen_ggml_lost(ctx); return -1; }
    report_sched(g, gf, logits ? (n > 1 ? "batch" : "decode") : "prefill");

    if (logits) {
        size_t need = (size_t)n * cfg->vocab_size;
        if (need > g->logits_cap) {
            free(g->logits);
            g->logits = (float *)malloc(need * sizeof(float));
            g->logits_cap = g->logits ? need : 0;
            if (!g->logits) { ggml_free(c); return -1; }
        }
        ggml_backend_tensor_get(logits, g->logits, 0, need * sizeof(float));
    }
    ggml_free(c);
    return 0;
}

const float *qwen_ggml_logits(qwen_ctx_t *ctx, int row) {
    gdec_t *g = (gdec_t *)ctx->ggml_dec;
    return g->logits + (size_t)row * ctx->config.vocab_size;
}

/* ========================================================================
 * Encoder: per-chunk conv stem, windowed attention, projector
 * ======================================================================== */

int qwen_ggml_encoder_init(qwen_ctx_t *ctx) {
    gdec_t *g = (gdec_t *)ctx->ggml_dec;
    const qwen_config_t *cfg = &ctx->config;
    qwen_encoder_t *enc = &ctx->encoder;
    if (!g) return -1;
    int d = cfg->enc_d_model, ffn = cfg->enc_ffn_dim, od = cfg->enc_output_dim;
    int ch = QWEN_CONV_HIDDEN, proj_in = QWEN_CONV_HIDDEN * 16;

    struct ggml_init_params p = { (size_t)(16 * cfg->enc_layers + 16) * ggml_tensor_overhead(), NULL, true };
    g->ectx = ggml_init(p);
    struct ggml_context *c = g->ectx;
    g->c1w = ggml_new_tensor_4d(c, GGML_TYPE_F16, 3, 3, 1, ch);
    g->c2w = ggml_new_tensor_4d(c, GGML_TYPE_F16, 3, 3, ch, ch);
    g->c3w = ggml_new_tensor_4d(c, GGML_TYPE_F16, 3, 3, ch, ch);
    g->c1b = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, ch, 1);
    g->c2b = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, ch, 1);
    g->c3b = ggml_new_tensor_4d(c, GGML_TYPE_F32, 1, 1, ch, 1);
    g->conv_out = ggml_new_tensor_2d(c, GGML_TYPE_F16, proj_in, d);
    g->ln_post_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
    g->ln_post_b = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
    g->p1w = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, d);
    g->p1b = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
    g->p2w = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, od);
    g->p2b = ggml_new_tensor_1d(c, GGML_TYPE_F32, od);
    for (int i = 0; i < cfg->enc_layers; i++) {
        g->enc[i].wq = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, d);
        g->enc[i].wk = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, d);
        g->enc[i].wv = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, d);
        g->enc[i].wo = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, d);
        g->enc[i].fc1 = ggml_new_tensor_2d(c, GGML_TYPE_F16, d, ffn);
        g->enc[i].fc2 = ggml_new_tensor_2d(c, GGML_TYPE_F16, ffn, d);
        g->enc[i].bq = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].bk = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].bv = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].bo = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].b1 = ggml_new_tensor_1d(c, GGML_TYPE_F32, ffn);
        g->enc[i].b2 = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].ln1_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].ln1_b = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].ln2_w = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
        g->enc[i].ln2_b = ggml_new_tensor_1d(c, GGML_TYPE_F32, d);
    }
    g->ebuf = ggml_backend_alloc_ctx_tensors(c, g->dev);
    if (!g->ebuf) return -1;

    int bad = upload_f32_as_f16(g->c1w, enc->conv1_weight) | upload_f32_as_f16(g->c2w, enc->conv2_weight) |
              upload_f32_as_f16(g->c3w, enc->conv3_weight) | upload_f32_as_f16(g->conv_out, enc->conv_out_weight) |
              upload_f32_as_f16(g->p1w, enc->proj1_weight) | upload_f32_as_f16(g->p2w, enc->proj2_weight);
    upload_f32(g->c1b, enc->conv1_bias);
    upload_f32(g->c2b, enc->conv2_bias);
    upload_f32(g->c3b, enc->conv3_bias);
    upload_f32(g->ln_post_w, enc->ln_post_weight);
    upload_f32(g->ln_post_b, enc->ln_post_bias);
    upload_f32(g->p1b, enc->proj1_bias);
    upload_f32(g->p2b, enc->proj2_bias);
    for (int i = 0; i < cfg->enc_layers && !bad; i++) {
        qwen_enc_layer_t *l = &enc->layers[i];
        bad |= upload_f32_as_f16(g->enc[i].wq, l->wq_weight) | upload_f32_as_f16(g->enc[i].wk, l->wk_weight) |
               upload_f32_as_f16(g->enc[i].wv, l->wv_weight) | upload_f32_as_f16(g->enc[i].wo, l->wo_weight) |
               upload_f32_as_f16(g->enc[i].fc1, l->fc1_weight) | upload_f32_as_f16(g->enc[i].fc2, l->fc2_weight);
        upload_f32(g->enc[i].bq, l->wq_bias);
        upload_f32(g->enc[i].bk, l->wk_bias);
        upload_f32(g->enc[i].bv, l->wv_bias);
        upload_f32(g->enc[i].bo, l->wo_bias);
        upload_f32(g->enc[i].b1, l->fc1_bias);
        upload_f32(g->enc[i].b2, l->fc2_bias);
        upload_f32(g->enc[i].ln1_w, l->attn_norm_weight);
        upload_f32(g->enc[i].ln1_b, l->attn_norm_bias);
        upload_f32(g->enc[i].ln2_w, l->ffn_norm_weight);
        upload_f32(g->enc[i].ln2_b, l->ffn_norm_bias);
    }
    return bad ? -1 : 0;
}

static struct ggml_tensor *conv_stem(struct ggml_context *c, gdec_t *g, struct ggml_tensor *in) {
    struct ggml_tensor *x = ggml_gelu(c, ggml_add(c, ggml_conv_2d(c, g->c1w, in, 2, 2, 1, 1, 1, 1), g->c1b));
    x = ggml_gelu(c, ggml_add(c, ggml_conv_2d(c, g->c2w, x, 2, 2, 1, 1, 1, 1), g->c2b));
    return ggml_gelu(c, ggml_add(c, ggml_conv_2d(c, g->c3w, x, 2, 2, 1, 1, 1, 1), g->c3b));
}

/* [w3, 16, 480, N] stem output -> [d, w3 * N] tokens with per-chunk position embeddings */
static struct ggml_tensor *stem_tokens(struct ggml_context *c, gdec_t *g, struct ggml_tensor *s,
                                       struct ggml_tensor *pe, int d) {
    int w3 = (int)s->ne[0], n = (int)s->ne[3];
    struct ggml_tensor *t = ggml_cont(c, ggml_permute(c, s, 2, 0, 1, 3));   /* [16, 480, w3, N] */
    t = ggml_reshape_3d(c, t, s->ne[1] * s->ne[2], w3, n);
    t = ggml_mul_mat(c, g->conv_out, t);                                   /* [d, w3, N] */
    t = ggml_add(c, t, ggml_view_2d(c, pe, d, w3, pe->nb[1], 0));
    return ggml_reshape_2d(c, t, d, w3 * n);
}

static struct ggml_tensor *layer_norm(struct ggml_context *c, struct ggml_tensor *x,
                                      struct ggml_tensor *w, struct ggml_tensor *b) {
    return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), w), b);
}

static struct ggml_tensor *linear(struct ggml_context *c, struct ggml_tensor *w,
                                  struct ggml_tensor *b, struct ggml_tensor *x) {
    return ggml_add(c, ggml_mul_mat(c, w, x), b);
}

float *qwen_ggml_encoder_forward(qwen_ctx_t *ctx, const float *mel, int mel_frames, int *out_seq_len) {
    gdec_t *g = (gdec_t *)ctx->ggml_dec;
    const qwen_config_t *cfg = &ctx->config;
    int d = cfg->enc_d_model, hd = cfg->enc_head_dim, heads = cfg->enc_heads;
    int cs = cfg->enc_chunk_size;
    int n_full = mel_frames / cs, rem = mel_frames % cs;
    int tpc = ((((cs + 1) / 2) + 1) / 2 + 1) / 2;                      /* 13 tokens per full chunk */
    int w1 = (rem + 1) / 2, w2 = (w1 + 1) / 2, w3r = rem ? (w2 + 1) / 2 : 0;
    int T = n_full * tpc + w3r;
    if (T <= 0) return NULL;
    int W = tpc * (cfg->enc_n_window_infer / cs);                      /* attention window, tokens */
    int n_win = (T + W - 1) / W, Tp = n_win * W;

    struct ggml_init_params p = { g->meta_size, g->meta, true };
    struct ggml_context *c = ggml_init(p);
    struct ggml_cgraph *gf = ggml_new_graph_custom(c, GRAPH_MAX, false);

    struct ggml_tensor *pe = ggml_new_tensor_2d(c, GGML_TYPE_F32, d, tpc);
    ggml_set_input(pe);
    struct ggml_tensor *in_full = NULL, *in_part = NULL, *x = NULL;
    if (n_full > 0) {
        in_full = ggml_new_tensor_4d(c, GGML_TYPE_F32, cs, 128, 1, n_full);
        ggml_set_input(in_full);
        x = stem_tokens(c, g, conv_stem(c, g, in_full), pe, d);
    }
    if (rem > 0) {
        in_part = ggml_new_tensor_4d(c, GGML_TYPE_F32, rem, 128, 1, 1);
        ggml_set_input(in_part);
        struct ggml_tensor *xp = stem_tokens(c, g, conv_stem(c, g, in_part), pe, d);
        x = x ? ggml_concat(c, x, xp, 1) : xp;
    }

    /* Windows are attention batches; the padded tail of the last one is masked. */
    struct ggml_tensor *mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, W, W, 1, n_win);
    ggml_set_input(mask);
    float scale = 1.0f / sqrtf((float)hd);
    for (int i = 0; i < cfg->enc_layers; i++) {
        struct ggml_tensor *h = layer_norm(c, x, g->enc[i].ln1_w, g->enc[i].ln1_b);
        struct ggml_tensor *q = linear(c, g->enc[i].wq, g->enc[i].bq, h);
        struct ggml_tensor *k = linear(c, g->enc[i].wk, g->enc[i].bk, h);
        struct ggml_tensor *v = linear(c, g->enc[i].wv, g->enc[i].bv, h);
        if (Tp > T) {
            q = ggml_pad(c, q, 0, Tp - T, 0, 0);
            k = ggml_pad(c, k, 0, Tp - T, 0, 0);
            v = ggml_pad(c, v, 0, Tp - T, 0, 0);
        }
        q = ggml_permute(c, ggml_reshape_4d(c, q, hd, heads, W, n_win), 0, 2, 1, 3);
        k = ggml_permute(c, ggml_reshape_4d(c, k, hd, heads, W, n_win), 0, 2, 1, 3);
        v = ggml_permute(c, ggml_reshape_4d(c, v, hd, heads, W, n_win), 0, 2, 1, 3);
        struct ggml_tensor *a = ggml_flash_attn_ext(c, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        a = ggml_reshape_2d(c, a, d, Tp);
        if (Tp > T) a = ggml_view_2d(c, a, d, T, a->nb[1], 0);
        x = ggml_add(c, x, linear(c, g->enc[i].wo, g->enc[i].bo, a));

        h = layer_norm(c, x, g->enc[i].ln2_w, g->enc[i].ln2_b);
        h = ggml_gelu(c, linear(c, g->enc[i].fc1, g->enc[i].b1, h));
        x = ggml_add(c, x, linear(c, g->enc[i].fc2, g->enc[i].b2, h));
    }
    x = layer_norm(c, x, g->ln_post_w, g->ln_post_b);
    x = linear(c, g->p2w, g->p2b, ggml_gelu(c, linear(c, g->p1w, g->p1b, x)));
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(g->sched);
    if (!ggml_backend_sched_alloc_graph(g->sched, gf)) { ggml_free(c); return NULL; }

    float *pe_host = (float *)malloc((size_t)tpc * d * sizeof(float));
    float *buf = (float *)malloc((size_t)(n_full > 0 ? n_full : 1) * cs * 128 * sizeof(float));
    ggml_fp16_t *m = (ggml_fp16_t *)malloc((size_t)W * W * n_win * sizeof(ggml_fp16_t));
    float *out = (float *)malloc((size_t)T * cfg->enc_output_dim * sizeof(float));
    if (!pe_host || !buf || !m || !out) {
        free(pe_host); free(buf); free(m); free(out); ggml_free(c); return NULL;
    }
    qwen_sinusoidal_pe(pe_host, tpc, d);
    ggml_backend_tensor_set(pe, pe_host, 0, (size_t)tpc * d * sizeof(float));
    /* mel is [128][mel_frames]; chunks are [128][cs] slices of it */
    for (int k = 0; k < n_full; k++)
        for (int b = 0; b < 128; b++)
            memcpy(buf + ((size_t)k * 128 + b) * cs, mel + (size_t)b * mel_frames + (size_t)k * cs,
                   (size_t)cs * sizeof(float));
    if (in_full) ggml_backend_tensor_set(in_full, buf, 0, ggml_nbytes(in_full));
    if (in_part) {
        for (int b = 0; b < 128; b++)
            memcpy(buf + (size_t)b * rem, mel + (size_t)b * mel_frames + (size_t)n_full * cs,
                   (size_t)rem * sizeof(float));
        ggml_backend_tensor_set(in_part, buf, 0, ggml_nbytes(in_part));
    }
    const uint16_t neg_inf = 0xFC00, zero = 0;
    for (int w = 0; w < n_win; w++) {
        int valid = T - w * W < W ? T - w * W : W;
        for (int i = 0; i < W; i++)
            for (int j = 0; j < W; j++)
                memcpy(&m[((size_t)w * W + i) * W + j], j < valid ? &zero : &neg_inf, 2);
    }
    ggml_backend_tensor_set(mask, m, 0, ggml_nbytes(mask));

    int ok = compute(g, gf);
    if (ok) report_sched(g, gf, "encoder");
    if (ok) ggml_backend_tensor_get(x, out, 0, (size_t)T * cfg->enc_output_dim * sizeof(float));
    free(pe_host); free(buf); free(m);
    ggml_free(c);
    if (!ok) { free(out); qwen_ggml_lost(ctx); return NULL; }
    *out_seq_len = T;
    return out;
}

#endif /* USE_GGML */
