/*
 * qwen_asr_kernels.c - Math kernels for Qwen3-ASR inference
 * Adapted from voxtral-realtime project.
 */

#include "qwen_asr_kernels.h"
#include "qwen_asr_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define QWEN_MAX_THREADS 16

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[QWEN_MAX_THREADS - 1];
    int tids[QWEN_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void qwen_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > QWEN_MAX_THREADS) n = QWEN_MAX_THREADS;

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
#ifdef USE_OPENBLAS
    /* Threading is ours (see pool_sgemm). */
    openblas_set_num_threads(1);
#endif
    if (n <= 1) return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Thread pool: %d threads\n", n);
}

int qwen_get_num_cpus(void) {
#ifdef __APPLE__
    /* Performance cores only: with the static row split, a thread on an
     * efficiency core makes the others wait. With fewer than 4 of them the
     * idle efficiency cores would cost more, so use all cores. */
    int n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, NULL, 0) == 0 && n >= 4)
        return n;
    n = 0;
    len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#elif defined(_WIN32)
    int n = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

/* Dispatch work to all threads; main thread is tid=0 */
static void parallel_for(parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}

/* OpenBLAS runs single-threaded; large sgemm calls are split into tiles of C
 * that the pool claims dynamically. C = A * op(B), op(B) = B^T if trans_b. */
#ifdef USE_OPENBLAS
typedef struct {
    float *C;
    const float *A, *B;
    int M, N, K, lda, ldb, ldc, trans_b;
    int mb, nb, tiles_n, n_tiles;
    int next_tile;
} pool_sgemm_task_t;

static void pool_sgemm_worker(int tid, int n_threads, void *arg) {
    (void)tid; (void)n_threads;
    pool_sgemm_task_t *t = (pool_sgemm_task_t *)arg;
    for (;;) {
        int u = __atomic_fetch_add(&t->next_tile, 1, __ATOMIC_RELAXED);
        if (u >= t->n_tiles) break;
        int m0 = (u / t->tiles_n) * t->mb, n0 = (u % t->tiles_n) * t->nb;
        int m = t->M - m0 < t->mb ? t->M - m0 : t->mb;
        int n = t->N - n0 < t->nb ? t->N - n0 : t->nb;
        const float *B = t->trans_b ? t->B + (size_t)n0 * t->ldb : t->B + n0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, t->trans_b ? CblasTrans : CblasNoTrans,
                    m, n, t->K, 1.0f, t->A + (size_t)m0 * t->lda, t->lda,
                    B, t->ldb, 0.0f, t->C + (size_t)m0 * t->ldc + n0, t->ldc);
    }
}

static int sgemm_tiles(int M, int N, int mb, int nb) {
    return ((M + mb - 1) / mb) * ((N + nb - 1) / nb);
}

static void pool_sgemm(int trans_b, int M, int N, int K, const float *A, int lda,
                       const float *B, int ldb, float *C, int ldc) {
    int nt = tp.n_threads;
    if (nt <= 1 || (double)M * N * K < 4e6) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                    M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
        return;
    }
    /* 256x256 tiles, smaller until there are about 3 per thread. */
    int mb = 256, nb = 256;
    while (sgemm_tiles(M, N, mb, nb) < 3 * nt && nb > 64) nb /= 2;
    while (sgemm_tiles(M, N, mb, nb) < 3 * nt && mb > 32) mb /= 2;
    pool_sgemm_task_t task = {
        .C = C, .A = A, .B = B, .M = M, .N = N, .K = K,
        .lda = lda, .ldb = ldb, .ldc = ldc, .trans_b = trans_b,
        .mb = mb, .nb = nb, .tiles_n = (N + nb - 1) / nb,
        .n_tiles = sgemm_tiles(M, N, mb, nb), .next_tile = 0,
    };
    parallel_for(pool_sgemm_worker, &task);
}
#define BIG_SGEMM(trans_b, M, N, K, A, lda, B, ldb, C, ldc) \
    pool_sgemm(trans_b, M, N, K, A, lda, B, ldb, C, ldc)
#elif defined(USE_BLAS)
#define BIG_SGEMM(trans_b, M, N, K, A, lda, B, ldb, C, ldc) \
    cblas_sgemm(CblasRowMajor, CblasNoTrans, (trans_b) ? CblasTrans : CblasNoTrans, \
                M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc)
#endif

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

void qwen_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void qwen_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void qwen_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void qwen_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    BIG_SGEMM(1, M, N, K, A, K, B, K, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void qwen_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    BIG_SGEMM(1, seq_len, out_dim, in_dim, x, in_dim, W, in_dim, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void qwen_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    qwen_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
}

/* Reusable scratch buffer for bf16->f32 conversion */
static float *bf16_scratch = NULL;
static size_t bf16_scratch_cap = 0;

typedef struct {
    float *dst;
    const uint16_t *src;
    size_t n;
} bf16_convert_task_t;

static void bf16_convert_worker(int tid, int n_threads, void *arg) {
    bf16_convert_task_t *t = (bf16_convert_task_t *)arg;
    size_t chunk = ((t->n + n_threads - 1) / n_threads + 15) & ~(size_t)15;
    size_t start = (size_t)tid * chunk;
    size_t end = start + chunk < t->n ? start + chunk : t->n;
    if (start < end) bf16_to_f32_buf(t->dst + start, t->src + start, end - start);
}

/* Prefill converts each weight on every call. */
static void bf16_to_f32_parallel(float *dst, const uint16_t *src, size_t n) {
    if (tp.n_threads > 1 && n >= 65536) {
        bf16_convert_task_t task = { dst, src, n };
        parallel_for(bf16_convert_worker, &task);
    } else {
        bf16_to_f32_buf(dst, src, n);
    }
}

static float *bf16_get_scratch(size_t n) {
    if (n > bf16_scratch_cap) {
        free(bf16_scratch);
        bf16_scratch = (float *)malloc(n * sizeof(float));
        bf16_scratch_cap = bf16_scratch ? n : 0;
    }
    return bf16_scratch;
}

typedef struct {
    const uint16_t *src;
    size_t n;
    float *dst_f32;
} bf16_cache_entry_t;

static bf16_cache_entry_t *bf16_cache = NULL;
static int bf16_cache_len = 0;
static int bf16_cache_cap = 0;
static size_t bf16_cache_bytes = 0;
static size_t bf16_cache_limit_bytes = 0;
static int bf16_cache_limit_init = 0;

static void bf16_cache_init_limit(void) {
    if (bf16_cache_limit_init) return;
    bf16_cache_limit_init = 1;

    /* Default OFF. Override with QWEN_BF16_CACHE_MB=<n> to enable. */
    unsigned long long mb = 0;
    const char *env = getenv("QWEN_BF16_CACHE_MB");
    if (env && env[0] != '\0') {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) mb = v;
    }
    bf16_cache_limit_bytes = (size_t)(mb * 1024ULL * 1024ULL);

    if (qwen_verbose >= 2) {
        fprintf(stderr, "BF16 cache: limit=%llu MB\n", mb);
    }
}

static const float *bf16_get_cached_f32(const uint16_t *src, size_t n) {
    bf16_cache_init_limit();

    for (int i = 0; i < bf16_cache_len; i++) {
        if (bf16_cache[i].src == src && bf16_cache[i].n == n) {
            return bf16_cache[i].dst_f32;
        }
    }

    if (bf16_cache_limit_bytes == 0) return NULL;

    size_t bytes = n * sizeof(float);
    if (bytes > bf16_cache_limit_bytes) return NULL;
    if (bf16_cache_bytes + bytes > bf16_cache_limit_bytes) return NULL;

    float *dst = (float *)malloc(bytes);
    if (!dst) return NULL;
    bf16_to_f32_buf(dst, src, n);

    if (bf16_cache_len == bf16_cache_cap) {
        int new_cap = bf16_cache_cap > 0 ? bf16_cache_cap * 2 : 256;
        bf16_cache_entry_t *tmp = (bf16_cache_entry_t *)realloc(
            bf16_cache, (size_t)new_cap * sizeof(bf16_cache_entry_t));
        if (!tmp) {
            free(dst);
            return NULL;
        }
        bf16_cache = tmp;
        bf16_cache_cap = new_cap;
    }

    bf16_cache[bf16_cache_len].src = src;
    bf16_cache[bf16_cache_len].n = n;
    bf16_cache[bf16_cache_len].dst_f32 = dst;
    bf16_cache_len++;
    bf16_cache_bytes += bytes;
    return dst;
}

static const float *bf16_get_f32_view(const uint16_t *src, size_t n) {
    const float *cached = bf16_get_cached_f32(src, n);
    if (cached) return cached;

    float *scratch = bf16_get_scratch(n);
    if (!scratch) return NULL;
    bf16_to_f32_parallel(scratch, src, n);
    return scratch;
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    qwen_bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} matvec_task_t;

static void matvec_worker(int tid, int n_threads, void *arg) {
    matvec_task_t *t = (matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    bf16_matvec_fused(t->y + start, t->x,
                      t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL,
                      t->in_dim, end - start);
}

static void bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    matvec_task_t task = { y, x, W_bf16, bias, in_dim, out_dim };
    parallel_for(matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} qkv_matvec_task_t;

static void qkv_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_matvec_task_t *t = (qkv_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            bf16_matvec_fused(t->q + s, t->x,
                              t->Wq_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + s, t->x,
                              t->Wk_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + s, t->x,
                              t->Wv_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }
}

void qwen_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(q, x, Wq_bf16, NULL, in_dim, q_dim);
        bf16_matvec_fused(k, x, Wk_bf16, NULL, in_dim, kv_dim);
        bf16_matvec_fused(v, x, Wv_bf16, NULL, in_dim, kv_dim);
        return;
    }

    qkv_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    parallel_for(qkv_matvec_worker, &task);
}

void qwen_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    qwen_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

void qwen_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    qwen_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* Find argmax over a range of output rows [start, end).
 * Uses 2-row processing to amortize x vector loads (same as bf16_matvec_fused). */
static void argmax_bf16_range(const float *x, const uint16_t *W_bf16,
                               int in_dim, int start, int end,
                               int *best_out, float *best_val_out) {
    qwen_argmax_bf16_range_impl(x, W_bf16, in_dim, start, end, best_out, best_val_out);
}

typedef struct {
    const float *x;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} argmax_task_t;

static void argmax_worker(int tid, int n_threads, void *arg) {
    argmax_task_t *t = (argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_bf16_range(t->x, t->W_bf16, t->in_dim, start, end,
                      &t->best_idx[tid], &t->best_val[tid]);
}

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        int best;
        float best_val;
        argmax_bf16_range(x, W_bf16, in_dim, 0, out_dim, &best, &best_val);
        return best;
    }

    argmax_task_t task;
    task.x = x;
    task.W_bf16 = W_bf16;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    parallel_for(argmax_worker, &task);

    int best = task.best_idx[0];
    float best_val = task.best_val[0];
    for (int i = 1; i < tp.n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best_idx[i];
        }
    }
    return best;
}

int qwen_argmax_matvec_bf16_excl(const float *x, const uint16_t *W_bf16,
                                  int in_dim, int out_dim, int excl) {
    int b1 = -1, b2 = -1;
    float v1 = 0, v2 = 0;
    if (excl > 0) argmax_bf16_range(x, W_bf16, in_dim, 0, excl, &b1, &v1);
    if (excl + 1 < out_dim) argmax_bf16_range(x, W_bf16, in_dim, excl + 1, out_dim, &b2, &v2);
    if (b1 < 0) return b2;
    if (b2 < 0) return b1;
    return v2 > v1 ? b2 : b1;
}

/* ========================================================================
 * Multi-vector bf16 matvec (a few tokens in one pass over the weights)
 *
 * Each thread takes the same row range as the single-vector kernels above and
 * walks it in blocks of an even number of rows, applying every vector to a
 * block while it is in cache. Per-row arithmetic, including which rows the
 * NEON kernel pairs, is unchanged, so results are bitwise those of nx
 * single-vector calls.
 * ======================================================================== */

#define QWEN_MULTI_ROW_BLOCK 32

static void matvec_multi_range(float *y, int ldy, const float *x, int ldx,
                               const uint16_t *W, int in_dim, int start, int end, int nx) {
    for (int b0 = start; b0 < end; b0 += QWEN_MULTI_ROW_BLOCK) {
        int b1 = b0 + QWEN_MULTI_ROW_BLOCK < end ? b0 + QWEN_MULTI_ROW_BLOCK : end;
        for (int r = 0; r < nx; r++)
            bf16_matvec_fused(y + (size_t)r * ldy + b0, x + (size_t)r * ldx,
                              W + (size_t)b0 * in_dim, NULL, in_dim, b1 - b0);
    }
}

typedef struct {
    float *y[3];
    const uint16_t *W[3];
    int rows[3];
    int ldy[3];
    int n_seg;
    const float *x;
    int ldx;
    int in_dim;
    int nx;
} matvec_multi_task_t;

/* Segments split like matvec_worker (one segment) or qkv_matvec_worker (three). */
static void matvec_multi_worker(int tid, int n_threads, void *arg) {
    matvec_multi_task_t *t = (matvec_multi_task_t *)arg;
    int total = 0;
    for (int s = 0; s < t->n_seg; s++) total += t->rows[s];
    int chunk = (total + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk < total ? start + chunk : total;
    int base = 0;
    for (int s = 0; s < t->n_seg; s++) {
        int seg_end = base + t->rows[s];
        int a = start > base ? start - base : 0;
        int e = (end < seg_end ? end : seg_end) - base;
        if (a < e)
            matvec_multi_range(t->y[s], t->ldy[s], t->x, t->ldx, t->W[s],
                               t->in_dim, a, e, t->nx);
        base = seg_end;
    }
}

static void matvec_multi_run(matvec_multi_task_t *task) {
    if (tp.n_threads <= 1) matvec_multi_worker(0, 1, task);
    else parallel_for(matvec_multi_worker, task);
}

void qwen_matvec_bf16_multi(float *y, int ldy, const float *x, int ldx,
                            const uint16_t *W_bf16, int in_dim, int out_dim, int nx) {
    matvec_multi_task_t task = {
        .y = { y }, .W = { W_bf16 }, .rows = { out_dim }, .ldy = { ldy }, .n_seg = 1,
        .x = x, .ldx = ldx, .in_dim = in_dim, .nx = nx,
    };
    matvec_multi_run(&task);
}

void qwen_matvec_bf16_qkv_multi(float *q, float *k, float *v, const float *x, int ldx,
                                const uint16_t *Wq_bf16, const uint16_t *Wk_bf16,
                                const uint16_t *Wv_bf16,
                                int in_dim, int q_dim, int kv_dim, int nx) {
    matvec_multi_task_t task = {
        .y = { q, k, v }, .W = { Wq_bf16, Wk_bf16, Wv_bf16 },
        .rows = { q_dim, kv_dim, kv_dim }, .ldy = { q_dim, kv_dim, kv_dim }, .n_seg = 3,
        .x = x, .ldx = ldx, .in_dim = in_dim, .nx = nx,
    };
    matvec_multi_run(&task);
}

typedef struct {
    const float *x;
    int ldx;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int nx;
    int *best_idx;      /* [QWEN_MAX_THREADS][nx] */
    float *best_val;
} argmax_multi_task_t;

static void argmax_multi_worker(int tid, int n_threads, void *arg) {
    argmax_multi_task_t *t = (argmax_multi_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk < t->out_dim ? start + chunk : t->out_dim;
    int *bi = t->best_idx + (size_t)tid * t->nx;
    float *bv = t->best_val + (size_t)tid * t->nx;
    for (int r = 0; r < t->nx; r++) { bi[r] = 0; bv[r] = -1e30f; }
    /* Strict > across blocks in row order keeps the first maximum, as the
     * single-vector scan does. */
    for (int b0 = start; b0 < end; b0 += QWEN_MULTI_ROW_BLOCK) {
        int b1 = b0 + QWEN_MULTI_ROW_BLOCK < end ? b0 + QWEN_MULTI_ROW_BLOCK : end;
        for (int r = 0; r < t->nx; r++) {
            int b;
            float val;
            argmax_bf16_range(t->x + (size_t)r * t->ldx, t->W_bf16, t->in_dim, b0, b1, &b, &val);
            if (b0 == start || val > bv[r]) { bv[r] = val; bi[r] = b; }
        }
    }
}

void qwen_argmax_matvec_bf16_multi(const float *x, int ldx, const uint16_t *W_bf16,
                                   int in_dim, int out_dim, int nx, int *best) {
    int nt = tp.n_threads > 1 ? tp.n_threads : 1;
    int best_idx[QWEN_MAX_THREADS * QWEN_MULTI_MAX_X];
    float best_val[QWEN_MAX_THREADS * QWEN_MULTI_MAX_X];
    argmax_multi_task_t task = { x, ldx, W_bf16, in_dim, out_dim, nx, best_idx, best_val };
    if (nt == 1) argmax_multi_worker(0, 1, &task);
    else parallel_for(argmax_multi_worker, &task);
    for (int r = 0; r < nx; r++) {
        int b = best_idx[r];
        float v = best_val[r];
        for (int i = 1; i < nt; i++) {
            if (best_val[i * nx + r] > v) {
                v = best_val[i * nx + r];
                b = best_idx[i * nx + r];
            }
        }
        best[r] = b;
    }
}

void qwen_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N) {
    if (M == 1) {
        bf16_matvec_threaded(C, A, B_bf16, NULL, K, N);
    } else {
        size_t n = (size_t)N * K;
        const float *B_f32 = bf16_get_f32_view(B_bf16, n);
        if (!B_f32) return;
        qwen_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 */
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    BIG_SGEMM(0, c_out, spatial_out, patch_size, weight, patch_size, cols, spatial_out,
              out, spatial_out);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int s = 0; s < spatial_out; s++) {
            float sum = 0.0f;
            for (int p = 0; p < patch_size; p++) {
                sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
            }
            out[oc * spatial_out + s] = sum;
        }
    }
#endif

    free(cols);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void qwen_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

/* tanh-approximation GELU, 256 values at a time so the tanh can go through
 * vForce on Apple. */
static void gelu_range(float *x, int n) {
    float inner[256];
    for (int i = 0; i < n; i += 256) {
        int m = n - i < 256 ? n - i : 256;
        float *v = x + i;
        for (int j = 0; j < m; j++)
            inner[j] = 0.7978845608028654f * (v[j] + 0.044715f * v[j] * v[j] * v[j]);
#if defined(USE_BLAS) && defined(__APPLE__)
        vvtanhf(inner, inner, &m);
#else
        for (int j = 0; j < m; j++) inner[j] = tanhf(inner[j]);
#endif
        for (int j = 0; j < m; j++) v[j] = 0.5f * v[j] * (1.0f + inner[j]);
    }
}

typedef struct {
    float *x;
    int n;
} gelu_task_t;

static void gelu_worker(int tid, int n_threads, void *arg) {
    gelu_task_t *t = (gelu_task_t *)arg;
    int chunk = ((t->n + n_threads - 1) / n_threads + 255) & ~255;
    int start = tid * chunk;
    int end = start + chunk < t->n ? start + chunk : t->n;
    if (start < end) gelu_range(t->x + start, end - start);
}

void qwen_gelu(float *x, int n) {
    if (tp.n_threads > 1 && n >= 16384) {
        gelu_task_t task = { x, n };
        parallel_for(gelu_worker, &task);
    } else {
        gelu_range(x, n);
    }
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} swiglu_task_t;

static void swiglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /* In-place mode (decode seq=1): keep single-pass scalar to avoid alias hazards. */
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
        }
    }
}

void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        parallel_for(swiglu_worker, &task);
    } else {
        swiglu_worker(0, 1, &task);
    }
}

void qwen_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

static inline float qwen_dot_f32(const float *a, const float *b, int n) {
    return qwen_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void qwen_vec_scale_inplace(float *dst, float scale, int n) {
    qwen_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void qwen_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    qwen_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void qwen_vec_scale_add(float *dst, const float *src, float correction, int n) {
    qwen_vec_scale_add_impl(dst, src, correction, n);
}

/* With BLAS, attention over many queries is two sgemm calls per head,
 * claimed through an atomic counter so efficiency cores take fewer. */
#if defined(USE_BLAS) && (defined(__APPLE__) || defined(USE_OPENBLAS))
#define QWEN_GEMM_ATTENTION 1

static float *attn_scores = NULL;
static size_t attn_scores_cap = 0;

static float *attn_scores_get(size_t per_thread) {
    size_t need = (size_t)(tp.n_threads > 1 ? tp.n_threads : 1) * per_thread;
    if (need > attn_scores_cap) {
        free(attn_scores);
        attn_scores = (float *)malloc(need * sizeof(float));
        attn_scores_cap = attn_scores ? need : 0;
    }
    return attn_scores;
}

/* Softmax over row[0, n); row[n, len) is zeroed (masked keys). */
static void attn_softmax_row(float *row, int n, int len) {
    float mx = row[0];
    for (int j = 1; j < n; j++) if (row[j] > mx) mx = row[j];
    for (int j = 0; j < n; j++) row[j] -= mx;
#ifdef __APPLE__
    vvexpf(row, row, &n);
#else
    for (int j = 0; j < n; j++) row[j] = expf(row[j]);
#endif
    float sum = 0.0f;
    for (int j = 0; j < n; j++) sum += row[j];
    float inv = 1.0f / sum;
    for (int j = 0; j < n; j++) row[j] *= inv;
    for (int j = n; j < len; j++) row[j] = 0.0f;
}

typedef struct {
    float *out;
    const float *Q, *K, *V;
    int n_heads, head_dim;
    float scale;
    const int *window_starts;
    int n_windows;
    int max_win;
    int next_unit;
    float *scores;
} gemm_bidir_task_t;

static void gemm_bidir_worker(int tid, int n_threads, void *arg) {
    (void)n_threads;
    gemm_bidir_task_t *t = (gemm_bidir_task_t *)arg;
    int hidden = t->n_heads * t->head_dim;
    float *S = t->scores + (size_t)tid * t->max_win * t->max_win;
    int n_units = t->n_heads * t->n_windows;

    for (;;) {
        int u = __atomic_fetch_add(&t->next_unit, 1, __ATOMIC_RELAXED);
        if (u >= n_units) break;
        int h = u % t->n_heads, w = u / t->n_heads;
        int ws = t->window_starts[w];
        int len = t->window_starts[w + 1] - ws;
        if (len <= 0) continue;
        size_t off = (size_t)ws * hidden + (size_t)h * t->head_dim;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    len, len, t->head_dim, t->scale,
                    t->Q + off, hidden, t->K + off, hidden, 0.0f, S, len);
        for (int i = 0; i < len; i++) attn_softmax_row(S + (size_t)i * len, len, len);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    len, t->head_dim, len, 1.0f,
                    S, len, t->V + off, hidden, 0.0f, t->out + off, hidden);
    }
}

static int gemm_bidirectional_attention(float *out, const float *Q, const float *K,
                                        const float *V, int n_heads, int head_dim,
                                        float scale, const int *window_starts,
                                        int n_windows) {
    int max_win = 0;
    for (int w = 0; w < n_windows; w++) {
        int len = window_starts[w + 1] - window_starts[w];
        if (len > max_win) max_win = len;
    }
    if (max_win <= 0) return -1;
    float *scores = attn_scores_get((size_t)max_win * max_win);
    if (!scores) return -1;
    gemm_bidir_task_t task = {
        .out = out, .Q = Q, .K = K, .V = V,
        .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
        .window_starts = window_starts, .n_windows = n_windows,
        .max_win = max_win, .next_unit = 0, .scores = scores,
    };
    if (tp.n_threads > 1) parallel_for(gemm_bidir_worker, &task);
    else gemm_bidir_worker(0, 1, &task);
    return 0;
}

typedef struct {
    float *out;
    const float *Q, *K, *V;
    int seq_q, seq_k, n_heads, n_kv_heads, head_dim, q_offset;
    float scale;
    int next_head;
    float *scores;
} gemm_causal_task_t;

static void gemm_causal_worker(int tid, int n_threads, void *arg) {
    (void)n_threads;
    gemm_causal_task_t *t = (gemm_causal_task_t *)arg;
    int q_hidden = t->n_heads * t->head_dim;
    int kv_hidden = t->n_kv_heads * t->head_dim;
    int heads_per_kv = t->n_heads / t->n_kv_heads;
    float *S = t->scores + (size_t)tid * t->seq_q * t->seq_k;

    for (;;) {
        int h = __atomic_fetch_add(&t->next_head, 1, __ATOMIC_RELAXED);
        if (h >= t->n_heads) break;
        int kv_h = h / heads_per_kv;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    t->seq_q, t->seq_k, t->head_dim, t->scale,
                    t->Q + h * t->head_dim, q_hidden,
                    t->K + kv_h * t->head_dim, kv_hidden, 0.0f, S, t->seq_k);
        for (int i = 0; i < t->seq_q; i++) {
            int n = t->q_offset + i + 1;
            if (n > t->seq_k) n = t->seq_k;
            attn_softmax_row(S + (size_t)i * t->seq_k, n, t->seq_k);
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    t->seq_q, t->head_dim, t->seq_k, 1.0f,
                    S, t->seq_k, t->V + kv_h * t->head_dim, kv_hidden, 0.0f,
                    t->out + h * t->head_dim, q_hidden);
    }
}

static int gemm_causal_attention(float *out, const float *Q, const float *K, const float *V,
                                 int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                 int head_dim, float scale, int q_offset) {
    float *scores = attn_scores_get((size_t)seq_q * seq_k);
    if (!scores) return -1;
    gemm_causal_task_t task = {
        .out = out, .Q = Q, .K = K, .V = V,
        .seq_q = seq_q, .seq_k = seq_k, .n_heads = n_heads, .n_kv_heads = n_kv_heads,
        .head_dim = head_dim, .q_offset = q_offset, .scale = scale,
        .next_head = 0, .scores = scores,
    };
    if (tp.n_threads > 1) parallel_for(gemm_causal_worker, &task);
    else gemm_causal_worker(0, 1, &task);
    return 0;
}
#endif /* QWEN_GEMM_ATTENTION */

void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq __attribute__((unused)),
                                   int n_heads, int head_dim, float scale,
                                   const int *window_starts, int n_windows) {
#ifdef QWEN_GEMM_ATTENTION
    if (gemm_bidirectional_attention(out, Q, K, V, n_heads, head_dim, scale,
                                     window_starts, n_windows) == 0)
        return;
#endif
    int hidden = n_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + i * hidden + h * head_dim;
                float *o_row = out + i * hidden + h * head_dim;

                /* Online softmax */
                float max_score = -1e30f;
                float sum_exp = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = ws; j < we; j++) {
                    const float *k_row = K + j * hidden + h * head_dim;
                    const float *v_row = V + j * hidden + h * head_dim;

                    float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                    if (score > max_score) {
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        float wt = expf(score - max_score);
                        sum_exp += wt;
                        qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if (sum_exp > 0.0f) {
                    float inv_sum = 1.0f / sum_exp;
                    qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

static void qwen_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} causal_attn_task_t;

static void causal_attn_worker(int tid, int n_threads, void *arg) {
    causal_attn_task_t *t = (causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

/* Few queries (decode): split the keys into blocks covering all heads, so each
 * thread reads whole contiguous K/V rows. Block softmax partials are merged after. */
#define SPLITK_BLOCK 128

typedef struct {
    const float *Q, *K, *V;
    int seq_q, seq_k, n_heads, n_kv_heads, head_dim, q_offset;
    float scale;
    int n_blocks;
    int next_block;
    float *part;    /* [n_blocks][seq_q][n_heads][2 + head_dim]: max, sum, out */
} splitk_attn_task_t;

static void splitk_attn_worker(int tid, int n_threads, void *arg) {
    (void)tid; (void)n_threads;
    splitk_attn_task_t *t = (splitk_attn_task_t *)arg;
    int H = t->n_heads, D = t->head_dim;
    int g = H / t->n_kv_heads;
    int q_hidden = H * D, kv_hidden = t->n_kv_heads * D;
    float S[H * SPLITK_BLOCK];

    for (;;) {
        int b = __atomic_fetch_add(&t->next_block, 1, __ATOMIC_RELAXED);
        if (b >= t->n_blocks) break;
        int j0 = b * SPLITK_BLOCK;
        for (int i = 0; i < t->seq_q; i++) {
            float *P = t->part + ((size_t)b * t->seq_q + i) * H * (2 + D);
            int j1 = t->q_offset + i + 1;
            if (j1 > t->seq_k) j1 = t->seq_k;
            if (j1 > j0 + SPLITK_BLOCK) j1 = j0 + SPLITK_BLOCK;
            int len = j1 - j0;
            if (len <= 0) {
                for (int h = 0; h < H; h++) P[h * (2 + D) + 1] = 0.0f;
                continue;
            }
            const float *q = t->Q + (size_t)i * q_hidden;

            for (int j = 0; j < len; j++) {
                const float *k_row = t->K + (size_t)(j0 + j) * kv_hidden;
                for (int h = 0; h < H; h++)
                    S[h * SPLITK_BLOCK + j] =
                        qwen_dot_f32(q + h * D, k_row + (h / g) * D, D) * t->scale;
            }
            for (int h = 0; h < H; h++) {
                float *s = S + h * SPLITK_BLOCK;
                float mx = s[0];
                for (int j = 1; j < len; j++) if (s[j] > mx) mx = s[j];
                for (int j = 0; j < len; j++) s[j] = expf(s[j] - mx);
                float sum = 0.0f;
                for (int j = 0; j < len; j++) sum += s[j];
                float *ph = P + h * (2 + D);
                ph[0] = mx;
                ph[1] = sum;
                memset(ph + 2, 0, D * sizeof(float));
            }
            for (int j = 0; j < len; j++) {
                const float *v_row = t->V + (size_t)(j0 + j) * kv_hidden;
                for (int h = 0; h < H; h++)
                    qwen_vec_axpy_inplace(P + h * (2 + D) + 2, v_row + (h / g) * D,
                                          S[h * SPLITK_BLOCK + j], D);
            }
        }
    }
}

static float *splitk_part = NULL;
static size_t splitk_part_cap = 0;

static int splitk_causal_attention(float *out, const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                   int head_dim, float scale, int q_offset) {
    int n_blocks = (seq_k + SPLITK_BLOCK - 1) / SPLITK_BLOCK;
    size_t stride = (size_t)n_heads * (2 + head_dim);
    size_t need = (size_t)n_blocks * seq_q * stride;
    if (need > splitk_part_cap) {
        free(splitk_part);
        splitk_part = (float *)malloc(need * sizeof(float));
        splitk_part_cap = splitk_part ? need : 0;
        if (!splitk_part) return -1;
    }
    splitk_attn_task_t task = {
        .Q = Q, .K = K, .V = V,
        .seq_q = seq_q, .seq_k = seq_k, .n_heads = n_heads, .n_kv_heads = n_kv_heads,
        .head_dim = head_dim, .q_offset = q_offset, .scale = scale,
        .n_blocks = n_blocks, .next_block = 0, .part = splitk_part,
    };
    if (tp.n_threads > 1 && n_blocks > 1) parallel_for(splitk_attn_worker, &task);
    else splitk_attn_worker(0, 1, &task);

    int D = head_dim;
    for (int i = 0; i < seq_q; i++) {
        for (int h = 0; h < n_heads; h++) {
            float *o = out + ((size_t)i * n_heads + h) * D;
            float mx = -1e30f;
            for (int b = 0; b < n_blocks; b++) {
                const float *ph = splitk_part + ((size_t)b * seq_q + i) * stride + (size_t)h * (2 + D);
                if (ph[1] > 0.0f && ph[0] > mx) mx = ph[0];
            }
            float sum = 0.0f;
            memset(o, 0, D * sizeof(float));
            for (int b = 0; b < n_blocks; b++) {
                const float *ph = splitk_part + ((size_t)b * seq_q + i) * stride + (size_t)h * (2 + D);
                if (ph[1] <= 0.0f) continue;
                float w = expf(ph[0] - mx);
                sum += w * ph[1];
                qwen_vec_axpy_inplace(o, ph + 2, w, D);
            }
            qwen_vec_scale_inplace(o, 1.0f / sum, D);
        }
    }
    return 0;
}

void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
#ifdef QWEN_GEMM_ATTENTION
    if (seq_q >= 8 &&
        gemm_causal_attention(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads,
                              head_dim, scale, q_offset) == 0)
        return;
#endif
    if (seq_q < 8 && n_heads <= 64 &&
        splitk_causal_attention(out, Q, K, V, seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset) == 0)
        return;
    if (tp.n_threads > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        parallel_for(causal_attn_worker, &task);
        return;
    }

    qwen_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Duplicate for full head_dim */
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * NeoX split-half style:
     *   x1 = x[..., :half], x2 = x[..., half:]
     *   rotated = cat(-x2, x1)
     *   result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            int d = 0;
            for (; d + 16 <= half; d += 16) {
                __m512 x1 = _mm512_loadu_ps(vec + d);
                __m512 x2 = _mm512_loadu_ps(vec + half + d);
                /* RoPE cache duplicates cos/sin across halves. */
                __m512 cc = _mm512_loadu_ps(c + d);
                __m512 ss = _mm512_loadu_ps(sn + d);
                __m512 new1 = _mm512_fmsub_ps(x1, cc, _mm512_mul_ps(x2, ss));
                __m512 new2 = _mm512_fmadd_ps(x2, cc, _mm512_mul_ps(x1, ss));
                _mm512_storeu_ps(vec + d, new1);
                _mm512_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#elif defined(__AVX2__) && defined(__FMA__)
            int d = 0;
            for (; d + 8 <= half; d += 8) {
                __m256 x1 = _mm256_loadu_ps(vec + d);
                __m256 x2 = _mm256_loadu_ps(vec + half + d);
                __m256 cc = _mm256_loadu_ps(c + d);
                __m256 ss = _mm256_loadu_ps(sn + d);
                __m256 new1 = _mm256_fmsub_ps(x1, cc, _mm256_mul_ps(x2, ss));
                __m256 new2 = _mm256_fmadd_ps(x2, cc, _mm256_mul_ps(x1, ss));
                _mm256_storeu_ps(vec + d, new1);
                _mm256_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#else
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];           /* first half */
                float x2 = vec[half + d];    /* second half */
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#endif
        }
    }
}
