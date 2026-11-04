/*
 * qwen_asr_quantize.c - quickasr quantize: write an int8 copy of a model
 *
 * A safetensors file with the decoder's linear weights as int8 blocks; the
 * rest, including the bf16 embedding (also lm_head), is copied. Layout in
 * MODEL.md.
 */

#include "qwen_asr.h"
#include "qwen_asr_kernels.h"
#include "qwen_asr_safetensors.h"
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

#define MAX_ITEMS (SAFETENSORS_MAX_TENSORS * 2)

typedef struct {
    char name[300];
    const char *dtype;
    int elem;                /* bytes per element */
    int ndim;
    int64_t shape[8];
    const void *data;
    size_t size;
    size_t offset;
} item_t;

typedef struct {
    item_t *items;
    int n;
    void *owned[MAX_ITEMS];  /* buffers made here */
    int n_owned;
} plan_t;

static const char *dtype_name(safetensor_dtype_t t, int *elem) {
    switch (t) {
    case DTYPE_F32:  *elem = 4; return "F32";
    case DTYPE_F16:  *elem = 2; return "F16";
    case DTYPE_BF16: *elem = 2; return "BF16";
    case DTYPE_I32:  *elem = 4; return "I32";
    case DTYPE_I64:  *elem = 8; return "I64";
    case DTYPE_BOOL: *elem = 1; return "BOOL";
    case DTYPE_I8:   *elem = 1; return "I8";
    default:         return NULL;
    }
}

static item_t *add_item(plan_t *p, const char *name, const char *dtype, int elem,
                        int ndim, const int64_t *shape, const void *data, size_t size) {
    if (p->n >= MAX_ITEMS) return NULL;
    item_t *it = &p->items[p->n++];
    snprintf(it->name, sizeof(it->name), "%s", name);
    it->dtype = dtype;
    it->elem = elem;
    it->ndim = ndim;
    for (int i = 0; i < ndim && i < 8; i++) it->shape[i] = shape[i];
    it->data = data;
    it->size = size;
    return it;
}

static void *own(plan_t *p, void *buf) {
    if (buf) p->owned[p->n_owned++] = buf;
    return buf;
}

/* Add base.q8 + base.q8d for a f32 [rows, cols] matrix. */
static int add_q8(plan_t *p, const char *base, const float *src, int64_t rows, int64_t cols) {
    if (cols % QWEN_Q8_BLOCK != 0 || p->n_owned + 2 > MAX_ITEMS) return -1;
    size_t n = (size_t)rows * cols;
    int64_t dcols = cols / QWEN_Q8_BLOCK;
    size_t dn = (size_t)rows * dcols;
    int8_t *qs = own(p, malloc(n));
    float *d = own(p, malloc(dn * sizeof(float)));
    if (!qs || !d) return -1;
    qwen_q8_from_f32(qs, d, src, (int)rows, (int)cols);

    char name[300];
    int64_t sq[2] = { rows, cols }, sd[2] = { rows, dcols };
    snprintf(name, sizeof(name), "%s.q8", base);
    if (!add_item(p, name, "I8", 1, 2, sq, qs, n)) return -1;
    snprintf(name, sizeof(name), "%s.q8d", base);
    if (!add_item(p, name, "F32", 4, 2, sd, d, dn * sizeof(float))) return -1;
    return 0;
}

static float *tensor_f32(multi_safetensors_t *ms, const char *name, const safetensor_t **out) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t || t->ndim != 2) {
        fprintf(stderr, "quantize: no 2-D tensor %s\n", name);
        return NULL;
    }
    *out = t;
    float *f = safetensors_get_f32(sf, t);
    if (!f) fprintf(stderr, "quantize: cannot read %s (only F32 and BF16 are supported)\n", name);
    return f;
}

static int ends_with(const char *s, const char *suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

/* layers.<n>.<rest>.weight of the decoder: the part to quantize, or NULL */
static const char *linear_part(const char *name, size_t pfx_len, int *layer) {
    static const char *const parts[] = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    };
    const char *p = name + pfx_len;
    if (strncmp(p, "layers.", 7) != 0) return NULL;
    char *end;
    long n = strtol(p + 7, &end, 10);
    if (end == p + 7 || *end != '.') return NULL;
    *layer = (int)n;
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        size_t len = strlen(parts[i]);
        if (strncmp(end + 1, parts[i], len) == 0 && strcmp(end + 1 + len, ".weight") == 0)
            return parts[i];
    }
    return NULL;
}

static int add_linear(plan_t *p, multi_safetensors_t *ms, const char *name, const char *part,
                      const char *pfx, int layer) {
    char base[300];
    const safetensor_t *t = NULL;
    float *f = tensor_f32(ms, name, &t);
    if (!f) return -1;
    snprintf(base, sizeof(base), "%slayers.%d.%s", pfx, layer, part);
    int rc = add_q8(p, base, f, t->shape[0], t->shape[1]);
    free(f);
    return rc;
}

/* gate and up with their rows interleaved, as the decoder's fused matvec wants */
static int add_gate_up(plan_t *p, multi_safetensors_t *ms, const char *pfx, int layer) {
    char name[300], base[300];
    const safetensor_t *tg = NULL, *tu = NULL;
    snprintf(name, sizeof(name), "%slayers.%d.mlp.gate_proj.weight", pfx, layer);
    float *g = tensor_f32(ms, name, &tg);
    snprintf(name, sizeof(name), "%slayers.%d.mlp.up_proj.weight", pfx, layer);
    float *u = tensor_f32(ms, name, &tu);
    int rc = -1;
    if (g && u && tg->shape[0] == tu->shape[0] && tg->shape[1] == tu->shape[1]) {
        int64_t rows = tg->shape[0], cols = tg->shape[1];
        float *fused = malloc((size_t)(2 * rows) * cols * sizeof(float));
        if (fused) {
            size_t row_bytes = (size_t)cols * sizeof(float);
            for (int64_t r = 0; r < rows; r++) {
                memcpy(fused + (size_t)(2 * r) * cols, g + (size_t)r * cols, row_bytes);
                memcpy(fused + (size_t)(2 * r + 1) * cols, u + (size_t)r * cols, row_bytes);
            }
            snprintf(base, sizeof(base), "%slayers.%d.mlp.gate_up", pfx, layer);
            rc = add_q8(p, base, fused, 2 * rows, cols);
            free(fused);
        }
    }
    free(g);
    free(u);
    return rc;
}

/* The bf16 embedding table the decoder needs; F32 fine-tunes are converted. */
static int add_embedding(plan_t *p, multi_safetensors_t *ms, const char *name) {
    safetensors_file_t *sf = NULL;
    const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
    if (!t) return -1;
    if (t->dtype == DTYPE_BF16)
        return add_item(p, t->name, "BF16", 2, t->ndim, t->shape, safetensors_data(sf, t),
                        t->data_size) ? 0 : -1;
    if (t->dtype != DTYPE_F32) {
        fprintf(stderr, "quantize: embedding must be F32 or BF16\n");
        return -1;
    }
    size_t n = (size_t)safetensor_numel(t);
    uint16_t *b = own(p, malloc(n * sizeof(uint16_t)));
    if (!b) return -1;
    const float *f = (const float *)safetensors_data(sf, t);
    for (size_t i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &f[i], sizeof(u));
        b[i] = (uint16_t)(u >> 16);
    }
    return add_item(p, t->name, "BF16", 2, t->ndim, t->shape, b, n * sizeof(uint16_t)) ? 0 : -1;
}

static int build_plan(plan_t *p, multi_safetensors_t *ms, const char *pfx) {
    size_t pfx_len = strlen(pfx);
    int max_layer = -1;
    for (int s = 0; s < ms->num_shards; s++) {
        safetensors_file_t *sf = ms->shards[s];
        for (int i = 0; i < sf->num_tensors; i++) {
            const safetensor_t *t = &sf->tensors[i];
            int layer = 0;
            const char *part = NULL;
            if (ends_with(t->name, "lm_head.weight")) continue;  /* tied to the embedding */
            if (strncmp(t->name, pfx, pfx_len) == 0) {
                if (strcmp(t->name + pfx_len, "embed_tokens.weight") == 0) {
                    if (add_embedding(p, ms, t->name) != 0) return -1;
                    continue;
                }
                part = linear_part(t->name, pfx_len, &layer);
            }
            if (part) {
                if (layer > max_layer) max_layer = layer;
                if (strstr(part, "gate_proj") || strstr(part, "up_proj")) continue;
                if (add_linear(p, ms, t->name, part, pfx, layer) != 0) return -1;
                continue;
            }
            int elem = 0;
            const char *dt = dtype_name(t->dtype, &elem);
            if (!dt) {
                fprintf(stderr, "quantize: unsupported dtype in %s\n", t->name);
                return -1;
            }
            if (!add_item(p, t->name, dt, elem, t->ndim, t->shape, safetensors_data(sf, t),
                          t->data_size))
                return -1;
        }
    }
    for (int l = 0; l <= max_layer; l++)
        if (add_gate_up(p, ms, pfx, l) != 0) return -1;

    return 0;
}

static int cmp_items(const void *a, const void *b) {
    const item_t *x = a, *y = b;
    if (x->elem != y->elem) return y->elem - x->elem;
    return strcmp(x->name, y->name);
}

static int write_model(const plan_t *p, const char *path) {
    size_t cap = (size_t)p->n * 400 + 256, len = 0, off = 0;
    char *h = malloc(cap + 8);
    if (!h) return -1;
    len += (size_t)snprintf(h, cap, "{\"__metadata__\":{\"format\":\"pt\",\"quickasr_format\":\"%s\"}",
                            QWEN_Q8_FORMAT);
    for (int i = 0; i < p->n; i++) {
        item_t *it = &p->items[i];
        it->offset = off;
        len += (size_t)snprintf(h + len, cap - len, ",\"%s\":{\"dtype\":\"%s\",\"shape\":[",
                                it->name, it->dtype);
        for (int d = 0; d < it->ndim; d++)
            len += (size_t)snprintf(h + len, cap - len, "%s%lld", d ? "," : "",
                                    (long long)it->shape[d]);
        len += (size_t)snprintf(h + len, cap - len, "],\"data_offsets\":[%llu,%llu]}",
                                (unsigned long long)off, (unsigned long long)(off + it->size));
        off += it->size;
    }
    h[len++] = '}';
    while ((8 + len) % 8) h[len++] = ' ';

    FILE *f = fopen(path, "wb");
    if (!f) { free(h); return -1; }
    uint64_t hlen = len;
    int bad = fwrite(&hlen, 8, 1, f) != 1 || fwrite(h, 1, len, f) != len;
    free(h);
    for (int i = 0; i < p->n && !bad; i++)
        bad = fwrite(p->items[i].data, 1, p->items[i].size, f) != p->items[i].size;
    if (fclose(f) != 0) bad = 1;
    return bad ? -1 : 0;
}

/* config and tokenizer files go along with the weights */
static int copy_side_files(const char *model_dir, const char *out_dir) {
    DIR *d = opendir(model_dir);
    if (!d) return -1;
    struct dirent *e;
    int rc = 0;
    char src[4096], dst[4096], buf[65536];
    while (rc == 0 && (e = readdir(d))) {
        if (e->d_name[0] == '.' || ends_with(e->d_name, ".safetensors") ||
            ends_with(e->d_name, ".safetensors.index.json"))
            continue;
        snprintf(src, sizeof(src), "%s/%s", model_dir, e->d_name);
        snprintf(dst, sizeof(dst), "%s/%s", out_dir, e->d_name);
        struct stat st;
        if (stat(src, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        FILE *in = fopen(src, "rb"), *out = in ? fopen(dst, "wb") : NULL;
        if (!out) rc = -1;
        size_t n;
        while (out && (n = fread(buf, 1, sizeof(buf), in)) > 0)
            if (fwrite(buf, 1, n, out) != n) rc = -1;
        if (in) fclose(in);
        if (out && fclose(out) != 0) rc = -1;
    }
    closedir(d);
    return rc;
}

int qwen_quantize_model(const char *model_dir, const char *out_dir) {
    char path[4096], tmp[4200];
    snprintf(path, sizeof(path), "%s/model.safetensors", out_dir);
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
#ifdef _WIN32
    int mk = _mkdir(out_dir);
#else
    int mk = mkdir(out_dir, 0755);
#endif
    if (mk != 0 && errno != EEXIST) {
        fprintf(stderr, "quantize: cannot create %s\n", out_dir);
        return -1;
    }
    struct stat st;
    if (stat(path, &st) == 0) {
        fprintf(stderr, "quantize: %s already exists\n", path);
        return -1;
    }

    multi_safetensors_t *ms = multi_safetensors_open(model_dir);
    if (!ms) {
        fprintf(stderr, "quantize: cannot open safetensors in %s\n", model_dir);
        return -1;
    }
    const char *pfx = multi_safetensors_find(ms, "model.audio_tower.conv2d1.weight", NULL)
                          ? "model.language_model." : "thinker.model.";
    char name[300];
    snprintf(name, sizeof(name), "%slayers.0.self_attn.q_proj.q8", pfx);
    int rc = -1;
    plan_t *p = calloc(1, sizeof(plan_t));
    if (p) p->items = calloc(MAX_ITEMS, sizeof(item_t));
    if (multi_safetensors_find(ms, name, NULL)) {
        fprintf(stderr, "quantize: %s is already int8\n", model_dir);
    } else if (p && p->items && build_plan(p, ms, pfx) == 0) {
        qsort(p->items, (size_t)p->n, sizeof(item_t), cmp_items);
        if (write_model(p, tmp) == 0 && rename(tmp, path) == 0 &&
            copy_side_files(model_dir, out_dir) == 0) {
            rc = 0;
            if (qwen_verbose >= 1)
                fprintf(stderr, "Wrote %s (%s, %d tensors)\n", path, QWEN_Q8_FORMAT, p->n);
        } else {
            fprintf(stderr, "quantize: cannot write %s\n", out_dir);
            remove(tmp);
        }
    }
    if (p) {
        for (int i = 0; i < p->n_owned; i++) free(p->owned[i]);
        free(p->items);
        free(p);
    }
    multi_safetensors_close(ms);
    return rc;
}
