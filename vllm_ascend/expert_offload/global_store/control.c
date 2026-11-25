// SPDX-License-Identifier: Apache-2.0
// CPU-only, process-shared collectives. No ACL, Python, or background threads.
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define CONTROL_MAGIC UINT64_C(0x4558504354524c01)
#define CACHE_LINE 64
#define ERROR_BYTES 512
#define SPIN_NS UINT64_C(50000)
#define SLEEP_NS 20000
#define KIND_COUNTS 1
#define KIND_SCORES 2
#define KIND_READY 3

typedef struct {
    uint64_t sequence, acknowledged;
    int64_t layer;
    uint64_t kind, bytes;
    unsigned char padding[CACHE_LINE - 5 * sizeof(uint64_t)];
    uint64_t failed;
    unsigned char failure_padding[CACHE_LINE - sizeof(uint64_t)];
    char failure[ERROR_BYTES];
} Slot;

typedef struct {
    uint64_t magic, world, capacity, stride;
    unsigned char padding[CACHE_LINE - 4 * sizeof(uint64_t)];
} Header;

_Static_assert(sizeof(Header) == CACHE_LINE, "Header alignment");
_Static_assert(sizeof(Slot) % CACHE_LINE == 0, "Slot alignment");
_Static_assert(__atomic_always_lock_free(sizeof(uint64_t), 0), "Lock-free 64-bit atomics required");

static uint64_t load(const uint64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void store(uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
}
static Slot *slot(Header *h, uint64_t r) {
    return (Slot *)((unsigned char *)h + sizeof(Header) + r * h->stride);
}
static unsigned char *payload(Slot *s) { return (unsigned char *)s + sizeof(Slot); }

uint64_t expert_control_size(uint64_t world, uint64_t capacity) {
    if (!world || world > 1024 || !capacity || capacity > UINT64_C(1073741824)) return 0;
    uint64_t stride = (sizeof(Slot) + capacity + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
    return sizeof(Header) + world * stride;
}
int expert_control_init(void *base, uint64_t world, uint64_t capacity) {
    uint64_t size = expert_control_size(world, capacity);
    if (!base || !size) return -1;
    memset(base, 0, size);
    Header *h = base;
    h->world = world; h->capacity = capacity;
    h->stride = (sizeof(Slot) + capacity + CACHE_LINE - 1) / CACHE_LINE * CACHE_LINE;
    store(&h->magic, CONTROL_MAGIC);
    return 0;
}
int expert_control_validate(void *base, uint64_t world, uint64_t capacity) {
    Header *h = base;
    return load(&h->magic) == CONTROL_MAGIC && h->world == world && h->capacity == capacity ? 0 : -1;
}
void expert_control_abort(void *base, uint64_t rank, const char *message) {
    Header *h = base;
    Slot *s = slot(h, rank);
    uint64_t expected = 0;
    // A separate failure area is immutable after publication, including while
    // another thread's watchdog races with the collective's own deadline.
    if (__atomic_compare_exchange_n(&s->failed, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        snprintf(s->failure, sizeof(s->failure), "%s", message);
        store(&s->failed, 2);
    }
}
static int error(Header *h, uint64_t rank, const char *reason, char *out, uint64_t limit) {
    expert_control_abort(h, rank, reason);
    snprintf(out, limit, "%s", reason);
    return -1;
}
static int wait_all(Header *h, uint64_t rank, uint64_t sequence, int acknowledged,
                    uint64_t deadline, char *out, uint64_t limit) {
    uint64_t start = now_ns();
    for (;;) {
        int ready = 1;
        uint64_t missing_rank = rank;
        for (uint64_t r = 0; r < h->world; ++r) {
            Slot *s = slot(h, r);
            if (load(&s->failed) == 2) {
                snprintf(out, limit, "Shared expert control aborted by rank %llu: %s",
                         (unsigned long long)r, s->failure);
                return -1;
            }
            uint64_t seen = load(acknowledged ? &s->acknowledged : &s->sequence);
            if (seen < sequence) { ready = 0; missing_rank = r; }
            if (!acknowledged && seen > sequence)
                return error(h, rank, "Shared expert control sequence overtaken", out, limit);
        }
        if (ready) return 0;
        uint64_t current = now_ns();
        if (current >= deadline) {
            char reason[ERROR_BYTES];
            snprintf(reason, sizeof(reason),
                     "Shared expert control timed out waiting for peer %llu at sequence %llu (%s)",
                     (unsigned long long)missing_rank, (unsigned long long)sequence,
                     acknowledged ? "read acknowledgement" : "publication");
            return error(h, rank, reason, out, limit);
        }
        if (current - start >= SPIN_NS) {
            struct timespec delay = {0, SLEEP_NS};
            nanosleep(&delay, NULL);
        }
    }
}
static int begin(Header *h, uint64_t rank, uint64_t seq, int64_t layer, uint64_t kind,
                 uint64_t bytes, uint64_t deadline, char *out, uint64_t limit) {
    if (rank >= h->world || !seq || bytes > h->capacity) return -1;
    // A fast rank must not reuse a slot until every reader has consumed it.
    if (wait_all(h, rank, seq - 1, 1, deadline, out, limit)) return -1;
    Slot *s = slot(h, rank);
    s->layer = layer; s->kind = kind; s->bytes = bytes;
    return 0;
}
static int publish(Header *h, uint64_t rank, uint64_t seq, uint64_t deadline,
                   char *out, uint64_t limit) {
    Slot *own = slot(h, rank);
    store(&own->sequence, seq);
    if (wait_all(h, rank, seq, 0, deadline, out, limit)) return -1;
    for (uint64_t r = 0; r < h->world; ++r) {
        Slot *s = slot(h, r);
        if (s->layer != own->layer || s->kind != own->kind ||
            (own->kind != KIND_READY && s->bytes != own->bytes)) {
            char reason[ERROR_BYTES];
            snprintf(reason, sizeof(reason),
                     "Shared expert control operation mismatch at sequence %llu: rank %llu layer=%lld kind=%llu; "
                     "peer %llu layer=%lld kind=%llu",
                     (unsigned long long)seq, (unsigned long long)rank, (long long)own->layer,
                     (unsigned long long)own->kind, (unsigned long long)r,
                     (long long)s->layer, (unsigned long long)s->kind);
            return error(h, rank, reason, out, limit);
        }
    }
    return 0;
}
int expert_control_statistics(void *base, uint64_t rank, uint64_t seq, int64_t layer,
                              uint64_t count, const int64_t *counts, const float *scores,
                              int64_t *counts_out, float *scores_out, uint64_t timeout_ns,
                              char *out, uint64_t limit) {
    Header *h = base;
    uint64_t deadline = now_ns() + timeout_ns;
    uint64_t bytes = count * (sizeof(int64_t) + (scores ? sizeof(float) : 0));
    if (begin(h, rank, seq, layer, scores ? KIND_SCORES : KIND_COUNTS, bytes, deadline, out, limit)) return -1;
    unsigned char *local = payload(slot(h, rank));
    memcpy(local, counts, count * sizeof(int64_t));
    if (scores) memcpy(local + count * sizeof(int64_t), scores, count * sizeof(float));
    if (publish(h, rank, seq, deadline, out, limit)) return -1;
    memset(counts_out, 0, count * sizeof(int64_t));
    if (scores) memset(scores_out, 0, count * sizeof(float));
    // Fixed rank order keeps all local planner inputs identical.
    for (uint64_t r = 0; r < h->world; ++r) {
        const int64_t *c = (const int64_t *)payload(slot(h, r));
        const float *f = (const float *)(c + count);
        for (uint64_t i = 0; i < count; ++i) {
            counts_out[i] += c[i];
            if (scores) scores_out[i] += f[i];
        }
    }
    store(&slot(h, rank)->acknowledged, seq);
    return 0;
}
int expert_control_ready(void *base, uint64_t rank, uint64_t seq, int64_t layer,
                         const char *failure, uint64_t timeout_ns, char *out, uint64_t limit) {
    Header *h = base;
    uint64_t bytes = strlen(failure) + 1, deadline = now_ns() + timeout_ns;
    if (begin(h, rank, seq, layer, KIND_READY, bytes, deadline, out, limit)) return -1;
    memcpy(payload(slot(h, rank)), failure, bytes);
    if (publish(h, rank, seq, deadline, out, limit)) return -1;
    int failures = 0;
    size_t used = 0;
    out[0] = 0;
    for (uint64_t r = 0; r < h->world; ++r) {
        const char *message = (const char *)payload(slot(h, r));
        if (*message) {
            ++failures;
            if (used + 1 < limit) {
                int n = snprintf(out + used, limit - used, "%srank %llu: %s",
                                 used ? "; " : "", (unsigned long long)r, message);
                if (n > 0) used += (size_t)n < limit - used ? (size_t)n : limit - used - 1;
            }
        }
    }
    store(&slot(h, rank)->acknowledged, seq);
    return failures;
}
