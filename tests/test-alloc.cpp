#include "common.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <vector>

//
// dummy backend with configurable max_buffer_size, tracks allocations

uint8_t * const alloc_base = (uint8_t *) 16;

struct dummy_backend_context {
    size_t max_buffer_size = 64;
    size_t alignment       = 8;

    ggml_backend_buffer_i              buffer_interface;
    ggml_backend_device                device;
    ggml_backend                       backend;
    std::vector<ggml_backend_buffer_t> buffers;

    size_t allocated_total() const {
        size_t n = 0;
        for (ggml_backend_buffer_t buf : buffers) {
            n += ggml_backend_buffer_get_size(buf);
        }
        return n;
    }
};

// ggml_backend_buffer_type interface

static const char * dummy_backend_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "dummy_buffer_type";
}

static ggml_backend_buffer_t dummy_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    dummy_backend_context * ctx    = (dummy_backend_context *) buft->context;
    ggml_backend_buffer_t & buffer = ctx->buffers.emplace_back();
    buffer                         = ggml_backend_buffer_init(buft, ctx->buffer_interface, ctx, size);
    return buffer;
}

static size_t dummy_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->alignment;
}

static size_t dummy_backend_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    dummy_backend_context * ctx = (dummy_backend_context *) buft->context;
    return ctx->max_buffer_size;
}

static bool dummy_backend_buffer_type_is_host(ggml_backend_buffer_type_t) {
    return true;
}

// ggml_backend_buffer interface

static void dummy_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    dummy_backend_context * ctx = (dummy_backend_context *) buffer->context;

    auto i = std::find(ctx->buffers.begin(), ctx->buffers.end(), buffer);
    GGML_ASSERT(i != ctx->buffers.end());
    ctx->buffers.erase(i);
}

static void * dummy_backend_buffer_get_base(ggml_backend_buffer_t) {
    return alloc_base;
}

static ggml_status dummy_backend_buffer_init_tensor(ggml_backend_buffer_t, ggml_tensor *) {
    return GGML_STATUS_SUCCESS;
}

static void dummy_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor *, uint8_t, size_t, size_t) {}

static void dummy_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t) {}

static void dummy_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor *, void *, size_t, size_t) {}

static void dummy_backend_buffer_clear(ggml_backend_buffer_t, uint8_t) {}

// ggml_backend_device interface

static enum ggml_backend_dev_type dummy_backend_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_CPU;
}

static bool dummy_backend_device_supports_op(ggml_backend_dev_t, const ggml_tensor *) {
    return true;
}

static bool dummy_backend_device_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    return device->context == buft->context;
}

// ggml_backend interface

static const char * dummy_backend_get_name(ggml_backend_t) {
    return "dummy_backend";
}

// dummy_backend

struct dummy_backend {
    std::unique_ptr<dummy_backend_context> context;
    ggml_backend_buffer_type               buffer_type;
};

static dummy_backend dummy_backend_init(size_t max_buffer_size, size_t alignment = 8) {
    dummy_backend b{};
    b.context                  = std::make_unique<dummy_backend_context>();
    b.context->alignment       = alignment;
    b.context->max_buffer_size = max_buffer_size;

    b.context->buffer_interface.free_buffer   = dummy_backend_buffer_free_buffer;
    b.context->buffer_interface.get_base      = dummy_backend_buffer_get_base;
    b.context->buffer_interface.init_tensor   = dummy_backend_buffer_init_tensor;
    b.context->buffer_interface.memset_tensor = dummy_backend_buffer_memset_tensor;
    b.context->buffer_interface.set_tensor    = dummy_backend_buffer_set_tensor;
    b.context->buffer_interface.get_tensor    = dummy_backend_buffer_get_tensor;
    b.context->buffer_interface.clear         = dummy_backend_buffer_clear;

    b.context->device.context             = b.context.get();
    b.context->device.iface.get_type      = dummy_backend_device_get_type;
    b.context->device.iface.supports_op   = dummy_backend_device_supports_op;
    b.context->device.iface.supports_buft = dummy_backend_device_supports_buft;

    b.context->backend.context        = b.context.get();
    b.context->backend.device         = &b.context->device;
    b.context->backend.iface.get_name = dummy_backend_get_name;

    b.buffer_type.device              = &b.context->device;
    b.buffer_type.context             = b.context.get();
    b.buffer_type.iface.get_name      = dummy_backend_buffer_type_get_name;
    b.buffer_type.iface.alloc_buffer  = dummy_backend_buffer_type_alloc_buffer;
    b.buffer_type.iface.get_alignment = dummy_backend_buffer_type_get_alignment;
    b.buffer_type.iface.get_max_size  = dummy_backend_buffer_type_get_max_size;
    b.buffer_type.iface.is_host       = dummy_backend_buffer_type_is_host;
    return b;
}

//
// test utilities

struct test_context_with_graph {
    ggml_context *   ctx;
    ggml_cgraph *    graph;
    ggml_context_ptr ctx_ptr;
};

static test_context_with_graph make_context() {
    ggml_init_params params{};
    params.mem_size = 48 * ggml_tensor_overhead() + ggml_graph_overhead();
    params.no_alloc = true;

    ggml_context *   ctx     = ggml_init(params);
    ggml_context_ptr ctx_ptr = ggml_context_ptr(ctx);
    ggml_cgraph *    graph   = ggml_new_graph(ctx);
    return { ctx, graph, std::move(ctx_ptr) };
}

static ggml_tensor * make_input_1d(ggml_context * ctx, int64_t n_elements) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    ggml_set_input(t);
    return t;
}

static ggml_tensor * make_input_with_size(ggml_context * ctx, size_t size_bytes) {
    GGML_ASSERT(size_bytes % 4 == 0);
    return make_input_1d(ctx, size_bytes / 4);
}

static void assign_names(ggml_context * ctx, const char * prefix = "x") {
    int i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        ggml_format_name(t, "%s%d", prefix, i++);
    }
}

static int get_leaf_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_leafs; ++i) {
        if (strncmp(graph->leafs[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "leaf not found: %s\n", tensor_name);
    return -1;
}

static int get_node_id(ggml_cgraph * graph, const char * tensor_name) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (strncmp(graph->nodes[i]->name, tensor_name, GGML_MAX_NAME) == 0) {
            return i;
        }
    }
    fprintf(stderr, "node not found: %s", tensor_name);
    return -1;
}

static ggml_gallocr_ptr allocate_graph(ggml_cgraph * graph, ggml_tensor * out, ggml_backend_buffer_type_t buft) {
    ggml_set_output(out);
    ggml_build_forward_expand(graph, out);

    ggml_gallocr_ptr galloc = ggml_gallocr_ptr(ggml_gallocr_new(buft));
    bool             result = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(result);
    return galloc;
}

//
// correctness checks for result allocations

static void check_all_allocated(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        GGML_ASSERT(t->buffer != nullptr);
        GGML_ASSERT(t->data != nullptr);
    }
}

static void check_max_size(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        auto   buft     = ggml_backend_buffer_get_type(t->buffer);
        size_t max_size = ggml_backend_buft_get_max_size(buft);
        size_t offset   = (char *) t->data - (char *) ggml_backend_buffer_get_base(t->buffer);
        GGML_ASSERT(t->data >= ggml_backend_buffer_get_base(t->buffer));
        GGML_ASSERT((size_t) offset + ggml_nbytes(t) <= max_size);
    }
}

static bool can_reuse_memory(ggml_cgraph * graph, int current_i, ggml_tensor * current, ggml_tensor * other) {
    if (other->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    // Check if `other` is still "alive", ie. an input to any node after the `current` op
    for (int i = current_i; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * t = ggml_graph_node(graph, i);
        for (int s = 0; s < GGML_MAX_SRC; s++) {
            if (t == current && ggml_op_can_inplace(t->op)) {
                continue;
            }
            if (t->src[s] == other) {
                return false;
            }
            if (t->src[s] && t->src[s]->view_src == other) {
                return false;
            }
        }
    }
    return true;
}

static bool memory_overlap(ggml_tensor * a, ggml_tensor * b) {
    if (a->buffer != b->buffer) {
        return false;
    }
    int64_t a0 = (int64_t) a->data;
    int64_t a1 = a0 + ggml_nbytes(a);
    int64_t b0 = (int64_t) b->data;
    int64_t b1 = b0 + ggml_nbytes(b);
    return a1 > b0 && b1 > a0;
}

static ggml_tensor * get_view_source(ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return t;
}

static void check_no_overlap(ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        for (int j = 0; j < i; ++j) {
            ggml_tensor * t = ggml_graph_node(graph, i);
            ggml_tensor * o = ggml_graph_node(graph, j);
            GGML_ASSERT(t != o);

            if (get_view_source(t) == get_view_source(o)) {
                continue;
            }
            if (memory_overlap(t, o)) {
                GGML_ASSERT(can_reuse_memory(graph, i, t, o));
            }
        }
    }
}

//
// test cases

// Scenario where the first backend buffer is completely exhausted and there are further
// tensors which require a second buffer
static void test_max_size_too_many_tensors() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[7];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_mul(ctx, x[0], x[1]);
    x[4] = ggml_add(ctx, x[1], x[2]);
    x[5] = ggml_add(ctx, x[3], x[0]);
    x[6] = ggml_add(ctx, x[4], x[5]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[6], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 16 + 16);
}

// Scenario where there is some space left in the first buffer, but not enough to accommodate
// a larger tensor, so a second buffer is required
static void test_max_size_tensor_too_large() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 16);    // chunk 0, [0 , 16)
    x[1] = make_input_with_size(ctx, 8);     // chunk 0, [16, 24)
    x[2] = ggml_concat(ctx, x[0], x[1], 0);  // chunk 1, [0 , 24)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 24);
}

// Scenario where a single tensor exceeds the max buffer size - in this case the allocator
// should try to create a bigger buffer anyway, and wait for the backend to throw an error.
// Backends may report an artificially lower max size in some cases for compatibility reasons.
static void test_tensor_larger_than_max_size() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[1], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() == 24);
}

// This test assumes a max of 16 buffer chunks, and tries to allocate tensors that would
// require more. Expectation is that the last buffer should grow to fit everything,
// leaving it to the backend to error out if it can't allocate that much.
static void test_not_enough_chunks() {
    const int max_chunks = 16;
    const int max_size   = 8;

    dummy_backend backend      = dummy_backend_init(max_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[max_chunks + 1];
    for (int i = 0; i < max_chunks + 1; ++i) {
        x[i] = make_input_with_size(ctx, max_size);
    }
    ggml_tensor * acc = x[0];
    for (int i = 0; i < max_chunks; ++i) {
        acc = ggml_add(ctx, acc, x[i + 1]);
    }
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, acc, &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() > max_chunks * max_size);
}

// Fill up leftover unallocated space of a chunk after allocating a large tensor that
// requires a new chunk.
static void test_fill_leftover_space() {
    dummy_backend backend      = dummy_backend_init(16);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 8);
    x[1] = ggml_pad(ctx, x[0], 2, 0, 0, 0);
    x[3] = ggml_mean(ctx, x[1]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[3], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 12 + 16);
}

// Check that views don't require any extra memory
static void test_view_inplace() {
    dummy_backend backend      = dummy_backend_init(32);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[6];
    x[0] = make_input_1d(ctx, 4);                // chunk 0, [0, 16)
    x[1] = ggml_reshape_2d(ctx, x[0], 2, 2);     // view of x0
    x[2] = ggml_permute(ctx, x[1], 1, 0, 2, 3);  // view of x0
    x[3] = ggml_view_1d(ctx, x[2], 2, 4);        // view of x0
    x[4] = make_input_1d(ctx, 2);                // chunk 0, [16, 24)
    x[5] = ggml_add(ctx, x[3], x[4]);            // reuse (inplace add)
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[5], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 24);
}

static void test_reuse_and_free() {
    dummy_backend backend      = dummy_backend_init(40);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 24);
    x[1] = make_input_with_size(ctx, 8);
    x[2] = make_input_with_size(ctx, 8);
    x[3] = ggml_add(ctx, x[1], x[2]);        // reuse, free x2
    x[4] = ggml_pad(ctx, x[0], 2, 0, 0, 0);  // alloc new buffer, free x0
    x[5] = ggml_scale(ctx, x[4], 2.0f);      // alloc from free block
    x[6] = ggml_add(ctx, x[4], x[5]);        // reuse, free x5
    x[7] = ggml_view_1d(ctx, x[6], 2, 8);    // view
    x[8] = ggml_add(ctx, x[3], x[7]);        // reuse
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 40 + 32 + 32);
}

static void test_merge_free_block(size_t max_buffer_size) {
    dummy_backend backend      = dummy_backend_init(max_buffer_size);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = make_input_with_size(ctx, 16);
    x[2] = make_input_with_size(ctx, 16);
    x[3] = ggml_mean(ctx, x[0]);
    x[4] = ggml_mean(ctx, x[1]);
    x[5] = ggml_pad(ctx, x[2], 2, 0, 0, 0);
    x[6] = ggml_add(ctx, x[3], x[4]);
    x[7] = ggml_pad(ctx, x[6], 5, 0, 0, 0);
    x[8] = ggml_add(ctx, x[5], x[7]);
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[8], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend.context->allocated_total() <= 32 + 32 + 24);
}

// Check that previously allocated but freed memory is preferred over allocating
// additional memory, even if the remaining space in a chunk would match tensor size better
static void test_prefer_already_allocated_memory() {
    dummy_backend backend      = dummy_backend_init(32, /*align*/ 4);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[3];
    x[0] = make_input_with_size(ctx, 24);  // [24b][8b unused]
    x[1] = ggml_mean(ctx, x[0]);           // [24b free][4b][4b unused]
    x[2] = ggml_mean(ctx, x[1]);           // should be allocated in the 24b block
    assign_names(ctx);

    ggml_gallocr_ptr galloc = allocate_graph(graph, x[2], &backend.buffer_type);
    check_all_allocated(graph);
    check_no_overlap(graph);
    GGML_ASSERT(backend.context->allocated_total() <= 28);
}

// test for allocating on multiple devices with some tensors in the graph
// allocated externally (not by gallocr).
static void test_multiple_buffer_types() {
    dummy_backend backend_a = dummy_backend_init(32);
    dummy_backend backend_b = dummy_backend_init(SIZE_MAX);

    auto [ctx_a, _a, ctx_a_ptr] = make_context();
    auto [ctx_b, _b, ctx_b_ptr] = make_context();
    auto [ctx, graph, ctx_ptr]  = make_context();

    ggml_tensor * a[2];
    a[0] = make_input_with_size(ctx_a, 16);
    a[1] = make_input_with_size(ctx_a, 16);
    assign_names(ctx_a, "a");

    ggml_tensor * b[2];
    b[0] = make_input_with_size(ctx_b, 24);
    b[1] = make_input_with_size(ctx_b, 4);
    assign_names(ctx_b, "b");

    ggml_tensor * x[9];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_mul(ctx, x[0], a[0]);
    x[2] = ggml_pad(ctx, x[1], 2, 0, 0, 0);
    x[3] = ggml_mul(ctx, x[2], b[0]);
    x[4] = ggml_mean(ctx, x[3]);
    x[5] = ggml_add(ctx, x[4], b[1]);
    x[6] = ggml_pad(ctx, x[5], 3, 0, 0, 0);
    x[7] = ggml_add(ctx, x[6], a[1]);
    x[8] = ggml_scale(ctx, x[7], 2.0f);
    assign_names(ctx, "x");

    ggml_backend_buffer_ptr    buf_a(ggml_backend_alloc_ctx_tensors_from_buft(ctx_a, &backend_a.buffer_type));
    ggml_backend_buffer_ptr    buf_b(ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, &backend_b.buffer_type));
    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };

    // assign buffer types manually to avoid extra complexity from backend scheduler
    ggml_set_output(x[8]);
    ggml_build_forward_expand(graph, x[8]);

    GGML_ASSERT(graph->n_leafs == 5);
    int leaf_buffer_ids[5];
    leaf_buffer_ids[get_leaf_id(graph, "a0")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "a1")] = 0;
    leaf_buffer_ids[get_leaf_id(graph, "b0")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "b1")] = 1;
    leaf_buffer_ids[get_leaf_id(graph, "x0")] = 0;

    GGML_ASSERT(graph->n_nodes == 8);
    int node_buffer_ids[8];
    node_buffer_ids[get_node_id(graph, "x1")] = 0;
    node_buffer_ids[get_node_id(graph, "x2")] = 0;
    node_buffer_ids[get_node_id(graph, "x3")] = 1;
    node_buffer_ids[get_node_id(graph, "x4")] = 1;
    node_buffer_ids[get_node_id(graph, "x5")] = 1;
    node_buffer_ids[get_node_id(graph, "x6")] = 1;
    node_buffer_ids[get_node_id(graph, "x7")] = 0;
    node_buffer_ids[get_node_id(graph, "x8")] = 0;

    ggml_gallocr_ptr galloc(ggml_gallocr_new_n(bufts, 2));
    ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    ggml_gallocr_alloc_graph(galloc.get(), graph);

    check_all_allocated(graph);
    check_no_overlap(graph);
    check_max_size(ctx);
    GGML_ASSERT(backend_a.context->allocated_total() <= 32 + 32 + 24);
    GGML_ASSERT(backend_b.context->allocated_total() <= 32 + 24);
}

static void test_buffer_size_zero() {
    dummy_backend backend_a    = dummy_backend_init(SIZE_MAX);
    dummy_backend backend_b    = dummy_backend_init(SIZE_MAX);
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[2];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);

    ggml_set_output(x[1]);
    ggml_build_forward_expand(graph, x[1]);

    int leaf_buffer_ids[1] = { 0 };
    int node_buffer_ids[1] = { 0 };

    ggml_backend_buffer_type_t bufts[2] = { &backend_a.buffer_type, &backend_b.buffer_type };
    ggml_gallocr_ptr           galloc   = ggml_gallocr_ptr(ggml_gallocr_new_n(bufts, 2));
    bool                       res1     = ggml_gallocr_reserve_n(galloc.get(), graph, node_buffer_ids, leaf_buffer_ids);
    bool                       res2     = ggml_gallocr_alloc_graph(galloc.get(), graph);
    GGML_ASSERT(res1 && res2);

    check_all_allocated(graph);
    GGML_ASSERT(backend_a.context->allocated_total() == 16);
    GGML_ASSERT(backend_b.context->allocated_total() == 0);
}

// Test re-using gallocr for a different graph. The new graph has the same
// total size, but one of the chunks is larger, so reallocation is required.
static void test_reallocation() {
    dummy_backend    backend = dummy_backend_init(32, /*align*/ 4);
    ggml_gallocr_ptr galloc;
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[4];
        x[0] = make_input_with_size(ctx, 24);
        x[1] = make_input_with_size(ctx, 16);
        x[2] = ggml_view_1d(ctx, x[0], 4, 0);
        x[3] = ggml_add(ctx, x[2], x[1]);
        assign_names(ctx);

        galloc = allocate_graph(graph, x[3], &backend.buffer_type);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
    {
        auto [ctx, graph, ctx_ptr] = make_context();
        ggml_tensor * x[3];
        x[0] = make_input_with_size(ctx, 20);
        x[1] = make_input_with_size(ctx, 20);
        x[2] = ggml_add(ctx, x[0], x[1]);
        assign_names(ctx);
        ggml_set_output(x[2]);
        ggml_build_forward_expand(graph, x[2]);

        bool result = ggml_gallocr_alloc_graph(galloc.get(), graph);
        GGML_ASSERT(result);
        check_all_allocated(graph);
        GGML_ASSERT(backend.context->allocated_total() == 40);
    }
}

static void test_backend_graph_optimize(ggml_backend_t, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(graph->n_nodes == 3);
    params->add_alloc_dep(params->user_data, graph->nodes[0], graph->nodes[2]);
}

static bool graph_reuses_allocation(bool add_alloc_dep) {
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * x[4];
    x[0] = make_input_with_size(ctx, 16);
    x[1] = ggml_scale(ctx, x[0], 2.0f);
    x[2] = ggml_scale(ctx, x[1], 2.0f);
    x[3] = ggml_scale(ctx, x[2], 2.0f);

    ggml_set_output(x[3]);
    ggml_build_forward_expand(graph, x[3]);

    dummy_backend backend = dummy_backend_init(SIZE_MAX);
    if (add_alloc_dep) {
        backend.context->backend.iface.graph_optimize = test_backend_graph_optimize;
    }

    ggml_backend_t             backend_ptr = &backend.context->backend;
    ggml_backend_buffer_type_t buft        = &backend.buffer_type;
    ggml_backend_sched_ptr     sched(ggml_backend_sched_new(&backend_ptr, &buft, 1, 8, false, true));
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    return x[1]->data == x[2]->data;
}

static void test_graph_optimize_alloc_dep() {
    GGML_ASSERT(graph_reuses_allocation(false));
    GGML_ASSERT(!graph_reuses_allocation(true));
}

struct qsa_alloc_result {
    std::vector<int32_t> selected;
    bool reuses_score;
    int coalesced_dispatches;
};

static void qsa_alloc_log(enum ggml_log_level, const char * text, void * user_data) {
    int * dispatches = static_cast<int *>(user_data);
    if (strstr(text, "ggml_vulkan: topk-qsa-coalesced ")) {
        ++*dispatches;
    }
    fputs(text, stderr);
}

static qsa_alloc_result compute_qsa_alloc(ggml_backend_t backend, ggml_backend_t cpu, int64_t n_tokens, int64_t n_stream,
                                         bool identity_layout, bool force_alias) {
    const int64_t n_blocks = 1536;
    const int64_t n_kv = 4 * n_blocks;
    const int width = n_blocks;
    auto [ctx, graph, ctx_ptr] = make_context();

    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
            identity_layout ? n_tokens : n_blocks, identity_layout ? n_blocks : n_tokens, n_stream);
    ggml_tensor * cell_blk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
    ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_kv, n_tokens, n_stream);
    for (ggml_tensor * leaf : { input, cell_blk, mask }) {
        ggml_set_input(leaf);
        ggml_set_output(leaf);
    }

    ggml_tensor * score = ggml_scale(ctx, input, 2.0f);
    if (force_alias) {
        ggml_set_output(score);
    }
    ggml_tensor * score_view = identity_layout ? ggml_permute(ctx, score, 0, 1, 2, 3) :
                                               ggml_permute(ctx, score, 1, 0, 2, 3);
    ggml_tensor * a = ggml_cont(ctx, score_view);
    ggml_tensor * expanded = ggml_get_rows(ctx, a, cell_blk);
    expanded = ggml_cont(ctx, ggml_permute(ctx, expanded, 1, 0, 2, 3));
    ggml_tensor * mask_f32 = ggml_cast(ctx, mask, GGML_TYPE_F32);
    expanded = ggml_add(ctx, expanded, ggml_reshape_3d(ctx, mask_f32, n_kv, n_tokens, n_stream));
    ggml_tensor * selected = ggml_top_k(ctx, expanded, width);
    ggml_tensor * output = ggml_cast(ctx, selected, GGML_TYPE_F32);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    assign_names(ctx, "qsa_alloc_");

    ggml_backend_t backends[] = { backend, cpu };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, backend == cpu ? 1 : 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        GGML_ASSERT(ggml_backend_supports_op(backend, node));
        ggml_backend_sched_set_tensor_backend(sched.get(), node, backend);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), selected) == backend);
    if (force_alias) {
        // The ordinary graph is valid with this reuse; the expanded selection must reject it.
        GGML_ASSERT(ggml_nbytes(selected) == ggml_nbytes(score));
        GGML_ASSERT(!memory_overlap(output, score));
        selected->buffer = score->buffer;
        selected->data = score->data;
    }
    const bool reuses_score = memory_overlap(output, score);
    if (!force_alias && !identity_layout && n_tokens > 8 &&
            !getenv("GGML_VK_TOPK_QSA_COALESCED_DISABLE") &&
            strcmp(ggml_backend_name(backend), "CPU") != 0) {
        GGML_ASSERT(!memory_overlap(selected, score));
        GGML_ASSERT(reuses_score);
    }

    std::vector<float> scores(ggml_nelements(input));
    std::vector<int32_t> blocks(ggml_nelements(cell_blk));
    std::vector<ggml_fp16_t> masks(ggml_nelements(mask));
    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t i = 0; i < n_kv; ++i) {
            blocks[s * n_kv + i] = ((i / 4) * 5 + s * 7) % n_blocks;
        }
        for (int64_t t = 0; t < n_tokens; ++t) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                const int64_t offset = identity_layout ? t + n_tokens * (b + n_blocks * s) :
                                                        b + n_blocks * (t + n_tokens * s);
                scores[offset] = float(((13 * b + 17 * t + 19 * s) % n_blocks) * 4 + 1) / 2.0f;
            }
            const int64_t visible = n_kv - 4 * ((t + 3 * s) % 97);
            for (int64_t i = 0; i < n_kv; ++i) {
                const float bias = i < visible ? float(i % 4) * 0.125f : -std::numeric_limits<float>::infinity();
                masks[i + n_kv * (t + n_tokens * s)] = ggml_fp32_to_fp16(bias);
            }
        }
    }
    ggml_backend_tensor_set(input, scores.data(), 0, scores.size() * sizeof(float));
    ggml_backend_tensor_set(cell_blk, blocks.data(), 0, blocks.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask, masks.data(), 0, masks.size() * sizeof(ggml_fp16_t));

    int dispatches = 0;
    ggml_log_set(qsa_alloc_log, &dispatches);
    const ggml_status status = ggml_backend_sched_graph_compute(sched.get(), graph);
    ggml_backend_sched_synchronize(sched.get());
    ggml_log_set(nullptr, nullptr);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    std::vector<float> output_values(ggml_nelements(output));
    ggml_backend_tensor_get(output, output_values.data(), 0, output_values.size() * sizeof(float));
    std::vector<int32_t> result(output_values.size());
    for (size_t i = 0; i < result.size(); ++i) {
        GGML_ASSERT(output_values[i] >= 0.0f && output_values[i] < n_kv);
        result[i] = (int32_t) output_values[i];
        GGML_ASSERT(float(result[i]) == output_values[i]);
    }
    for (int64_t row = 0; row < n_tokens * n_stream; ++row) {
        auto begin = result.begin() + row * width;
        std::sort(begin, begin + width);
        GGML_ASSERT(std::adjacent_find(begin, begin + width) == begin + width);
    }
    return { std::move(result), reuses_score, dispatches };
}

static void test_vulkan_qsa_alloc_case(ggml_backend_t cpu, ggml_backend_t vulkan, int64_t n_tokens,
                                       int64_t n_stream, bool identity_layout, bool force_alias) {
    const auto reference = compute_qsa_alloc(cpu, cpu, n_tokens, n_stream, identity_layout, false);
    const auto result = compute_qsa_alloc(vulkan, cpu, n_tokens, n_stream, identity_layout, force_alias);
    GGML_ASSERT(result.selected == reference.selected);
    const bool expect_coalesced = n_tokens > 8 && !identity_layout && !force_alias &&
                                  !getenv("GGML_VK_TOPK_QSA_COALESCED_DISABLE");
    GGML_ASSERT(result.coalesced_dispatches == (expect_coalesced ? 1 : 0));
    printf("qsa_alloc tokens=%lld streams=%lld identity=%d alias=%d active=%d reuse=%d PASSED\n",
            (long long) n_tokens, (long long) n_stream, identity_layout, force_alias,
            result.coalesced_dispatches, result.reuses_score);
}

static int test_vulkan_qsa_alloc() {
    common_set_env("GGML_VK_TOPK_QSA_COALESCED_TRACE", "1");
    ggml_backend_load_all();
    ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
    ggml_backend_ptr vulkan(ggml_backend_init_by_name("Vulkan0", nullptr));
    GGML_ASSERT(cpu && vulkan);
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    test_vulkan_qsa_alloc_case(cpu.get(), vulkan.get(), 8, 1, false, false);
    test_vulkan_qsa_alloc_case(cpu.get(), vulkan.get(), 9, 1, false, false);
    test_vulkan_qsa_alloc_case(cpu.get(), vulkan.get(), 17, 2, false, false);
    test_vulkan_qsa_alloc_case(cpu.get(), vulkan.get(), 17, 1, false, true);
    test_vulkan_qsa_alloc_case(cpu.get(), vulkan.get(), 17, 1, true, false);
    return 0;
}

enum conv_input_variant {
    CONV_INPUT_NORMAL,
    CONV_INPUT_BITS,
    CONV_INPUT_MISALIGNED,
    CONV_INPUT_PAD_X,
    CONV_INPUT_PAD_HISTORY,
    CONV_INPUT_PERMUTE,
    CONV_INPUT_F16,
    CONV_INPUT_SWAPPED,
    CONV_INPUT_DIM1,
    CONV_INPUT_CONT_OUTPUT,
    CONV_INPUT_EXTRA_USE,
    CONV_INPUT_ALIAS,
};

struct conv_input_case {
    const char * name;
    int64_t tokens = 64;
    int64_t channels = 10240;
    int64_t history = 3;
    int64_t sequences = 1;
    int slots = 0;
    conv_input_variant variant = CONV_INPUT_NORMAL;
};

struct conv_input_result {
    std::array<std::vector<uint8_t>, 2> output;
    std::array<std::vector<uint8_t>, 2> cache;
    std::array<std::vector<uint8_t>, 2> exposed;
    bool reused = false;
};

static void conv_input_log(enum ggml_log_level, const char * text, void * user_data) {
    if (strstr(text, "ggml_vulkan: conv-input-direct ")) {
        ++*static_cast<int *>(user_data);
    }
    fputs(text, stderr);
}

static std::vector<uint8_t> conv_input_data(ggml_tensor * tensor, int phase, bool history, bool special) {
    const uint32_t words[] = { 0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu, 0x7F800000u, 0xFF800000u,
                              0x7FC12345u, 0xFFA01234u, 0x3F800001u, 0xBF000007u, 0x00800000u, 0x7F7FFFFFu };
    const size_t count = ggml_nelements(tensor);
    const size_t bytes = ggml_type_size(tensor->type);
    std::vector<uint8_t> data(count * bytes);
    for (size_t i = 0; i < count; ++i) {
        if (tensor->type == GGML_TYPE_F32) {
            const uint32_t bits = special ? words[(i * 5 + phase * 3 + (history ? 7 : 0)) % 12] :
                    (0x3F000001u + ((i * 65537 + phase * 17 + (history ? 113 : 0)) & 0x007FFFFEu)) | (history ? 0x80000000u : 0);
            memcpy(data.data() + i * bytes, &bits, bytes);
        } else {
            const ggml_fp16_t value = ggml_fp32_to_fp16((history ? -1.0f : 1.0f) * float(1 + (i + phase * 11) % 1023) / 1024.0f);
            memcpy(data.data() + i * bytes, &value, bytes);
        }
    }
    return data;
}

static conv_input_result compute_conv_input(ggml_backend_t backend, ggml_backend_t cpu, const conv_input_case & c, bool expect_active) {
    auto [ctx, graph, ctx_ptr] = make_context();
    const ggml_type type = c.variant == CONV_INPUT_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    const size_t unit = ggml_type_size(type);
    const size_t offset = c.variant == CONV_INPUT_MISALIGNED ? 2 * unit : 0;
    const int64_t x_row = c.channels + (c.variant == CONV_INPUT_PAD_X ? 1 : 0);
    const int64_t h_ne0 = c.variant == CONV_INPUT_DIM1 ? c.tokens : c.history;
    const int64_t h_ne1 = c.variant == CONV_INPUT_DIM1 ? c.history : c.channels;
    const int64_t h_row = h_ne0 + (c.variant == CONV_INPUT_PAD_HISTORY ? 1 : 0);
    ggml_tensor * x_input = ggml_new_tensor_1d(ctx, type, x_row * (c.tokens + c.history) * c.sequences + 32);
    ggml_tensor * h_input = ggml_new_tensor_1d(ctx, type, h_row * h_ne1 * c.sequences + 32);
    for (ggml_tensor * input : { x_input, h_input }) {
        ggml_set_input(input);
        ggml_set_output(input);
    }
    const bool generated = c.variant != CONV_INPUT_BITS;
    ggml_tensor * x_backing = generated ? ggml_dup(ctx, x_input) : x_input;
    ggml_tensor * h_backing = generated ? ggml_dup(ctx, h_input) : h_input;
    ggml_tensor * x = ggml_view_3d(ctx, x_backing, c.channels, c.tokens, c.sequences,
            x_row * unit, x_row * c.tokens * unit, offset);
    ggml_tensor * h = ggml_view_3d(ctx, h_backing, h_ne0, h_ne1, c.sequences,
            h_row * unit, h_row * h_ne1 * unit, offset);
    ggml_tensor * transposed = c.variant == CONV_INPUT_PERMUTE ? ggml_permute(ctx, x, 1, 0, 2, 3) : ggml_transpose(ctx, x);
    ggml_tensor * cont = ggml_cont(ctx, transposed);
    ggml_tensor * joined = c.variant == CONV_INPUT_SWAPPED ? ggml_concat(ctx, cont, h, 0) :
            ggml_concat(ctx, h, cont, c.variant == CONV_INPUT_DIM1 ? 1 : 0);
    if (c.variant == CONV_INPUT_CONT_OUTPUT) {
        ggml_set_output(cont);
    }
    ggml_build_forward_expand(graph, joined);
    ggml_tensor * exposed = nullptr;
    if (c.variant == CONV_INPUT_EXTRA_USE) {
        exposed = ggml_dup(ctx, cont);
        ggml_set_output(exposed);
        ggml_build_forward_expand(graph, exposed);
    } else if (c.variant == CONV_INPUT_CONT_OUTPUT) {
        exposed = cont;
    }

    ggml_tensor * cache = nullptr;
    const size_t slot_bytes = c.history * c.channels * c.sequences * unit;
    if (c.slots > 0) {
        cache = ggml_new_tensor_1d(ctx, type, c.slots * slot_bytes / unit + 16);
        ggml_set_input(cache);
        ggml_set_output(cache);
        for (int slot = 0; slot < c.slots; ++slot) {
            ggml_tensor * tail = ggml_view_3d(ctx, joined, c.history, c.channels, c.sequences,
                    joined->nb[1], joined->nb[2], std::max<int64_t>(0, c.tokens - slot) * unit);
            ggml_tensor * saved = ggml_view_2d(ctx, cache, c.history * c.channels, c.sequences,
                    c.history * c.channels * unit, 8 * unit + slot * slot_bytes);
            ggml_build_forward_expand(graph, ggml_cpy(ctx, ggml_cont(ctx, tail), saved));
        }
    }
    ggml_tensor * output = ggml_dup(ctx, joined);
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    assign_names(ctx, "conv_input_");

    ggml_backend_buffer_ptr external(nullptr);
    ggml_tensor * guard = nullptr;
    if (c.variant == CONV_INPUT_MISALIGNED) {
        guard = ggml_new_tensor_1d(ctx, type, ggml_nelements(joined) + 32);
        external.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), ggml_nbytes(guard)));
        GGML_ASSERT(external);
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(external.get()));
        GGML_ASSERT(ggml_backend_tensor_alloc(external.get(), guard, base) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_tensor_alloc(external.get(), joined, base + 5 * unit) == GGML_STATUS_SUCCESS);
    } else if (c.variant == CONV_INPUT_ALIAS) {
        external.reset(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), ggml_nbytes(x_backing)));
        GGML_ASSERT(external);
        void * base = ggml_backend_buffer_get_base(external.get());
        GGML_ASSERT(ggml_backend_tensor_alloc(external.get(), x_backing, base) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_tensor_alloc(external.get(), joined, base) == GGML_STATUS_SUCCESS);
    }

    ggml_backend_t backends[] = { backend, cpu };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, backend == cpu ? 1 : 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
    for (ggml_tensor * input : { x_input, h_input, cache }) {
        if (input) {
            ggml_backend_sched_set_tensor_backend(sched.get(), input, backend);
        }
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        GGML_ASSERT(ggml_backend_supports_op(backend, node));
        ggml_backend_sched_set_tensor_backend(sched.get(), node, backend);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), joined) == backend);
    if (expect_active) {
        GGML_ASSERT(!memory_overlap(joined, transposed));
        GGML_ASSERT(!memory_overlap(joined, h));
    }
    conv_input_result result;
    result.reused = memory_overlap(output, x_backing);
    if (expect_active && c.variant == CONV_INPUT_NORMAL && c.tokens == 64) {
        GGML_ASSERT(result.reused);
    }

    for (int phase = 0; phase < 2; ++phase) {
        auto x_data = conv_input_data(x_input, phase, false, c.variant == CONV_INPUT_BITS);
        auto h_data = conv_input_data(h_input, phase, true, c.variant == CONV_INPUT_BITS);
        if (phase == 1 && cache) {
            const int rollback = std::min(c.slots - 1, 2);
            memcpy(h_data.data() + offset, result.cache[0].data() + 8 * unit + rollback * slot_bytes, slot_bytes);
        }
        ggml_backend_tensor_set(x_input, x_data.data(), 0, x_data.size());
        ggml_backend_tensor_set(h_input, h_data.data(), 0, h_data.size());
        std::vector<uint8_t> poison(ggml_nbytes(output), 0xA5);
        ggml_backend_tensor_set(output, poison.data(), 0, poison.size());
        if (cache) {
            std::vector<uint8_t> sentinel(ggml_nbytes(cache), 0xA5);
            ggml_backend_tensor_set(cache, sentinel.data(), 0, sentinel.size());
        }
        if (guard) {
            std::vector<uint8_t> sentinel(ggml_nbytes(guard), 0xA5);
            ggml_backend_tensor_set(guard, sentinel.data(), 0, sentinel.size());
        }
        int dispatches = 0;
        ggml_log_set(conv_input_log, &dispatches);
        const ggml_status status = ggml_backend_sched_graph_compute(sched.get(), graph);
        ggml_backend_sched_synchronize(sched.get());
        ggml_log_set(nullptr, nullptr);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        GGML_ASSERT(dispatches == (expect_active ? 1 : 0));
        result.output[phase].resize(ggml_nbytes(output));
        ggml_backend_tensor_get(output, result.output[phase].data(), 0, result.output[phase].size());
        if (cache) {
            result.cache[phase].resize(ggml_nbytes(cache));
            ggml_backend_tensor_get(cache, result.cache[phase].data(), 0, result.cache[phase].size());
            GGML_ASSERT(std::all_of(result.cache[phase].begin(), result.cache[phase].begin() + 8 * unit, [](uint8_t b) { return b == 0xA5; }));
            GGML_ASSERT(std::all_of(result.cache[phase].end() - 8 * unit, result.cache[phase].end(), [](uint8_t b) { return b == 0xA5; }));
        }
        if (exposed) {
            result.exposed[phase].resize(ggml_nbytes(exposed));
            ggml_backend_tensor_get(exposed, result.exposed[phase].data(), 0, result.exposed[phase].size());
        }
        if (guard) {
            std::vector<uint8_t> sentinels(ggml_nbytes(guard));
            ggml_backend_tensor_get(guard, sentinels.data(), 0, sentinels.size());
            GGML_ASSERT(std::all_of(sentinels.begin(), sentinels.begin() + 5 * unit, [](uint8_t b) { return b == 0xA5; }));
            GGML_ASSERT(std::all_of(sentinels.end() - 27 * unit, sentinels.end(), [](uint8_t b) { return b == 0xA5; }));
        }
    }
    GGML_ASSERT(result.output[0] != result.output[1]);
    return result;
}

static int test_vulkan_conv_input(const char * filter) {
    common_set_env("GGML_VK_CONV_INPUT_DIRECT_TRACE", "1");
    ggml_backend_load_all();
    ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
    ggml_backend_ptr vulkan(ggml_backend_init_by_name("Vulkan0", nullptr));
    GGML_ASSERT(cpu && vulkan);
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    const bool disabled = getenv("GGML_VK_CONV_INPUT_DIRECT_DISABLE") || getenv("GGML_VK_DISABLE_FUSION") || getenv("GGML_VK_DISABLE_GRAPH_OPTIMIZE");
    const conv_input_case cases[] = {
        { "active64", 64, 10240, 3, 1, 1 },
        { "active65", 65, 10240, 3, 1, 3 },
        { "active127", 127, 10240, 3, 1, 4 },
        { "active2048", 2048, 10240, 3, 1, 4 },
        { "bits", 65, 10240, 3, 1, 0, CONV_INPUT_BITS },
        { "misaligned", 65, 10240, 3, 1, 3, CONV_INPUT_MISALIGNED },
        { "small", 63 },
        { "channels", 64, 10241 },
        { "history2", 64, 10240, 2 },
        { "history4", 64, 10240, 4 },
        { "sequences", 64, 10240, 3, 2 },
        { "padded_x", 64, 10240, 3, 1, 0, CONV_INPUT_PAD_X },
        { "padded_history", 64, 10240, 3, 1, 0, CONV_INPUT_PAD_HISTORY },
        { "permuted", 64, 10240, 3, 1, 0, CONV_INPUT_PERMUTE },
        { "f16", 64, 10240, 3, 1, 0, CONV_INPUT_F16 },
        { "swapped", 64, 10240, 3, 1, 0, CONV_INPUT_SWAPPED },
        { "dimension1", 64, 10240, 3, 1, 0, CONV_INPUT_DIM1 },
        { "cont_output", 64, 10240, 3, 1, 0, CONV_INPUT_CONT_OUTPUT },
        { "extra_use", 64, 10240, 3, 1, 0, CONV_INPUT_EXTRA_USE },
        { "alias", 64, 10240, 3, 1, 0, CONV_INPUT_ALIAS },
    };
    int selected = 0;
    for (const auto & c : cases) {
        if (filter && strcmp(filter, c.name) != 0) {
            continue;
        }
        ++selected;
        fprintf(stderr, "conv_input begin case=%s tokens=%lld slots=%d\n", c.name, (long long) c.tokens, c.slots);
        const bool eligible = c.tokens >= 64 && c.channels == 10240 && c.history == 3 && c.sequences == 1 &&
                (c.variant == CONV_INPUT_NORMAL || c.variant == CONV_INPUT_BITS || c.variant == CONV_INPUT_MISALIGNED);
        const auto reference = compute_conv_input(cpu.get(), cpu.get(), c, false);
        const auto result = compute_conv_input(vulkan.get(), cpu.get(), c, eligible && !disabled);
        GGML_ASSERT(result.output == reference.output);
        GGML_ASSERT(result.cache == reference.cache);
        GGML_ASSERT(result.exposed == reference.exposed);
        printf("conv_input case=%s tokens=%lld slots=%d active=%d reuse=%d phases=2 PASSED\n",
                c.name, (long long) c.tokens, c.slots, eligible && !disabled, result.reused);
    }
    GGML_ASSERT(selected > 0);
    return 0;
}

enum qsa_relu_keep {
    QSA_RELU_KEEP_NONE,
    QSA_RELU_KEEP_CHILD,
    QSA_RELU_KEEP_VIEW,
    QSA_RELU_KEEP_OUTPUT,
};

struct qsa_relu_shape {
    int64_t blocks;
    int64_t heads;
    int64_t tokens;
    int64_t streams;
    qsa_relu_keep keep = QSA_RELU_KEEP_NONE;
    bool dense = false;
};

struct qsa_relu_graph {
    test_context_with_graph memory;
    ggml_tensor * keys;
    ggml_tensor * queries;
    ggml_tensor * product;
    ggml_tensor * rectified;
    ggml_tensor * output;
    ggml_tensor * retained;
};

static qsa_relu_graph make_qsa_relu_graph(const qsa_relu_shape & shape, bool relu_first) {
    qsa_relu_graph result{};
    result.memory = make_context();
    ggml_context * ctx = result.memory.ctx;
    result.keys = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, shape.blocks, shape.streams);
    result.queries = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, shape.heads * shape.tokens, shape.streams);
    for (ggml_tensor * leaf : { result.keys, result.queries }) {
        ggml_set_input(leaf);
        ggml_set_output(leaf);
    }
    result.product = ggml_mul_mat(ctx, result.keys, result.queries);
    ggml_tensor * score = result.product;
    if (relu_first) {
        result.rectified = ggml_relu(ctx, score);
        score = ggml_reshape_4d(ctx, result.rectified, shape.blocks, shape.heads, shape.tokens, shape.streams);
    } else {
        score = ggml_reshape_4d(ctx, score, shape.blocks, shape.heads, shape.tokens, shape.streams);
        result.rectified = score = ggml_relu(ctx, score);
    }
    for (int64_t h = 0; h < shape.heads; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx, score, shape.blocks, shape.tokens, shape.streams,
                score->nb[2], score->nb[3], h * score->nb[1]);
        result.output = result.output ? ggml_add(ctx, result.output, slice) : ggml_cont(ctx, slice);
    }
    ggml_set_output(result.output);
    ggml_build_forward_expand(result.memory.graph, result.output);
    if (shape.keep == QSA_RELU_KEEP_CHILD || shape.keep == QSA_RELU_KEEP_VIEW) {
        ggml_tensor * raw = result.product;
        if (shape.keep == QSA_RELU_KEEP_VIEW) {
            raw = ggml_view_3d(ctx, raw, shape.blocks, shape.heads * shape.tokens, shape.streams,
                    raw->nb[1], raw->nb[2], 0);
        }
        // Keep signed, pre-ReLU values alive until after all head slices have been read.
        result.retained = ggml_dup(ctx, raw);
        ggml_set_output(result.retained);
        ggml_build_forward_expand(result.memory.graph, result.retained);
    } else if (shape.keep == QSA_RELU_KEEP_OUTPUT) {
        result.retained = result.product;
        ggml_set_output(result.retained);
    }
    assign_names(ctx, relu_first ? "qsa_relu_new_" : "qsa_relu_old_");
    return result;
}

static void place_qsa_relu_graph(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        GGML_ASSERT(ggml_backend_supports_op(backend, node));
        ggml_backend_sched_set_tensor_backend(sched, node, backend);
    }
}

static int qsa_relu_query_numerator(int64_t k, int64_t head, int64_t token, int64_t stream, int phase) {
    return int((7 * k + 3 * head + 5 * token + 11 * stream + 13 * phase) % 63) - 31;
}

static int qsa_relu_key_numerator(int64_t k, int64_t block, int64_t stream, int phase, bool dense) {
    return dense ? int((3 * k + 5 * block + 11 * stream + phase) % 7) - 3 : int(k == block % 128) * 8;
}

static void measure_qsa_relu_integer_error(
        ggml_backend_t backend, const qsa_relu_shape & shape, bool relu_first, int phase, const char * role,
        const std::vector<float> & actual, const std::vector<float> & expected) {
    GGML_ASSERT(actual.size() == expected.size() && !actual.empty());
    size_t bit_differences = 0, numeric_differences = 0, nonfinite_elements = 0;
    double max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        uint32_t actual_bits, expected_bits;
        memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
        memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
        numeric_differences += actual[i] != expected[i];
        if (std::isfinite(actual[i]) && std::isfinite(expected[i])) {
            max_abs = std::max(max_abs, std::fabs(double(actual[i]) - double(expected[i])));
        } else {
            ++nonfinite_elements;
            max_abs = std::numeric_limits<double>::infinity();
        }
        if (actual_bits != expected_bits) {
            if (bit_differences < 16) {
                printf("qsa_relu_compare_value backend=%s blocks=%lld heads=%lld tokens=%lld streams=%lld keep=%d dense=%d first=%d phase=%d role=%s index=%zu actual=%.9g actual_bits=%08x expected=%.9g expected_bits=%08x\n",
                        ggml_backend_name(backend), (long long) shape.blocks, (long long) shape.heads,
                        (long long) shape.tokens, (long long) shape.streams, int(shape.keep), shape.dense,
                        relu_first, phase, role, i, double(actual[i]), unsigned(actual_bits),
                        double(expected[i]), unsigned(expected_bits));
            }
            ++bit_differences;
        }
    }
    printf("qsa_relu_compare_metric backend=%s blocks=%lld heads=%lld tokens=%lld streams=%lld keep=%d dense=%d first=%d phase=%d role=%s integer_oracle_exact=%d bit_elements=%zu numeric_elements=%zu nonfinite_elements=%zu total_elements=%zu max_abs=%.17g\n",
            ggml_backend_name(backend), (long long) shape.blocks, (long long) shape.heads,
            (long long) shape.tokens, (long long) shape.streams, int(shape.keep), shape.dense,
            relu_first, phase, role, bit_differences == 0, bit_differences, numeric_differences,
            nonfinite_elements, actual.size(), max_abs);
    // Matching nonfinite results are still invalid evidence of arithmetic preservation.
    GGML_ASSERT(nonfinite_elements == 0);
}

static std::vector<uint8_t> compute_qsa_relu_order(
        ggml_backend_t backend, ggml_backend_t cpu, const qsa_relu_shape & shape, bool relu_first,
        bool compare_orders) {
    qsa_relu_shape maximum = shape;
    maximum.tokens = std::max<int64_t>(65, shape.tokens);
    auto reserve = make_qsa_relu_graph(maximum, relu_first);
    auto runtime = make_qsa_relu_graph(shape, relu_first);
    ggml_backend_t backends[] = { backend, cpu };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, backend == cpu ? 1 : 2,
            GGML_DEFAULT_GRAPH_SIZE, false, true));
    place_qsa_relu_graph(sched.get(), backend, reserve.memory.graph);
    GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), reserve.memory.graph));
    const size_t reserved = ggml_backend_sched_get_buffer_size(sched.get(), backend);
    const size_t reserved_cpu = ggml_backend_sched_get_buffer_size(sched.get(), cpu);
    for (int i = 0; i < ggml_graph_n_nodes(reserve.memory.graph); ++i) {
        GGML_ASSERT(ggml_graph_node(reserve.memory.graph, i)->data == nullptr);
    }
    place_qsa_relu_graph(sched.get(), backend, runtime.memory.graph);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), runtime.memory.graph));
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), backend) == reserved);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched.get(), cpu) == reserved_cpu);
    for (int i = 0; i < ggml_graph_n_nodes(runtime.memory.graph); ++i) {
        GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), ggml_graph_node(runtime.memory.graph, i)) == backend);
    }
    check_all_allocated(runtime.memory.graph);
    check_no_overlap(runtime.memory.graph);
    const bool reuse = runtime.product->buffer == runtime.rectified->buffer &&
                       runtime.product->data == runtime.rectified->data;
    GGML_ASSERT(reuse == (relu_first && shape.keep == QSA_RELU_KEEP_NONE));
    if (!reuse) {
        GGML_ASSERT(!memory_overlap(runtime.product, runtime.rectified));
    }

    std::vector<uint8_t> result;
    for (int phase : { 0, 1 }) {
        std::vector<float> keys(ggml_nelements(runtime.keys));
        std::vector<float> queries(ggml_nelements(runtime.queries));
        std::vector<float> expected(ggml_nelements(runtime.output), 0.0f);
        std::vector<float> raw(ggml_nelements(runtime.product));
        for (int64_t s = 0; s < shape.streams; ++s) {
            for (int64_t b = 0; b < shape.blocks; ++b) {
                for (int64_t k = 0; k < 128; ++k) {
                    keys[k + 128 * (b + shape.blocks * s)] = qsa_relu_key_numerator(k, b, s, phase, shape.dense) / 8.0f;
                }
            }
            for (int64_t t = 0; t < shape.tokens; ++t) {
                for (int64_t h = 0; h < shape.heads; ++h) {
                    for (int64_t k = 0; k < 128; ++k) {
                        queries[k + 128 * (h + shape.heads * (t + shape.tokens * s))] =
                                qsa_relu_query_numerator(k, h, t, s, phase) / 8.0f;
                    }
                    for (int64_t b = 0; b < shape.blocks; ++b) {
                        // The inputs are exact in F16. Integer products and sums give an
                        // independent F32 oracle without floating-point reduction rounding.
                        int dot = 0;
                        for (int64_t k = 0; k < 128; ++k) {
                            dot += qsa_relu_key_numerator(k, b, s, phase, shape.dense) *
                                   qsa_relu_query_numerator(k, h, t, s, phase);
                        }
                        const float value = dot / 64.0f;
                        raw[b + shape.blocks * (h + shape.heads * (t + shape.tokens * s))] = value;
                        expected[b + shape.blocks * (t + shape.tokens * s)] += std::max(value, 0.0f);
                    }
                }
            }
        }
        ggml_backend_tensor_set(runtime.keys, keys.data(), 0, ggml_nbytes(runtime.keys));
        ggml_backend_tensor_set(runtime.queries, queries.data(), 0, ggml_nbytes(runtime.queries));
        ggml_backend_tensor_memset(runtime.output, 0xa5, 0, ggml_nbytes(runtime.output));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), runtime.memory.graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(runtime.output, actual.data(), 0, ggml_nbytes(runtime.output));
        if (!compare_orders && memcmp(actual.data(), expected.data(), ggml_nbytes(runtime.output)) != 0) {
            size_t bit_differences = 0;
            size_t numeric_differences = 0;
            size_t nonfinite_differences = 0;
            double max_abs = 0.0;
            fprintf(stderr, "qsa_relu_mismatch backend=%s blocks=%lld heads=%lld tokens=%lld streams=%lld keep=%d dense=%d first=%d phase=%d\n",
                    ggml_backend_name(backend), (long long) shape.blocks, (long long) shape.heads,
                    (long long) shape.tokens, (long long) shape.streams, int(shape.keep), shape.dense, relu_first, phase);
            for (size_t i = 0; i < actual.size(); ++i) {
                uint32_t actual_bits;
                uint32_t expected_bits;
                memcpy(&actual_bits, &actual[i], sizeof(actual_bits));
                memcpy(&expected_bits, &expected[i], sizeof(expected_bits));
                if (actual_bits == expected_bits) {
                    continue;
                }
                numeric_differences += actual[i] != expected[i];
                if (std::isfinite(actual[i]) && std::isfinite(expected[i])) {
                    max_abs = std::max(max_abs, std::fabs(double(actual[i]) - double(expected[i])));
                } else {
                    ++nonfinite_differences;
                    max_abs = std::numeric_limits<double>::infinity();
                }
                if (bit_differences < 16) {
                    fprintf(stderr, "qsa_relu_value index=%zu actual=%.9g actual_bits=%08x expected=%.9g expected_bits=%08x\n",
                            i, double(actual[i]), unsigned(actual_bits), double(expected[i]), unsigned(expected_bits));
                }
                ++bit_differences;
            }
            fprintf(stderr, "qsa_relu_difference bit_elements=%zu numeric_elements=%zu nonfinite_elements=%zu total_elements=%zu max_abs=%.17g\n",
                    bit_differences, numeric_differences, nonfinite_differences, actual.size(), max_abs);
        }
        if (compare_orders) {
            measure_qsa_relu_integer_error(backend, shape, relu_first, phase, "output", actual, expected);
        } else {
            GGML_ASSERT(memcmp(actual.data(), expected.data(), ggml_nbytes(runtime.output)) == 0);
        }
        const uint8_t * bytes = reinterpret_cast<const uint8_t *>(actual.data());
        result.insert(result.end(), bytes, bytes + ggml_nbytes(runtime.output));
        if (runtime.retained) {
            std::vector<float> retained(raw.size());
            ggml_backend_tensor_get(runtime.retained, retained.data(), 0, ggml_nbytes(runtime.retained));
            if (compare_orders) {
                measure_qsa_relu_integer_error(backend, shape, relu_first, phase, "retained", retained, raw);
            } else {
                GGML_ASSERT(memcmp(retained.data(), raw.data(), ggml_nbytes(runtime.retained)) == 0);
            }
            bytes = reinterpret_cast<const uint8_t *>(retained.data());
            result.insert(result.end(), bytes, bytes + ggml_nbytes(runtime.retained));
        }
        // Input exports must survive both in-place evaluation and graph replay.
        std::vector<float> unchanged(keys.size());
        ggml_backend_tensor_get(runtime.keys, unchanged.data(), 0, ggml_nbytes(runtime.keys));
        GGML_ASSERT(memcmp(unchanged.data(), keys.data(), ggml_nbytes(runtime.keys)) == 0);
        unchanged.resize(queries.size());
        ggml_backend_tensor_get(runtime.queries, unchanged.data(), 0, ggml_nbytes(runtime.queries));
        GGML_ASSERT(memcmp(unchanged.data(), queries.data(), ggml_nbytes(runtime.queries)) == 0);
    }
    if (compare_orders) {
        printf("qsa_relu_compare_order backend=%s blocks=%lld heads=%lld tokens=%lld streams=%lld keep=%d dense=%d first=%d reuse=%d reserve_bytes=%zu phases=2 input_checks=2 allocation_checks=1 COMPLETE\n",
                ggml_backend_name(backend), (long long) shape.blocks, (long long) shape.heads,
                (long long) shape.tokens, (long long) shape.streams, int(shape.keep), shape.dense, relu_first, reuse, reserved);
    } else {
        printf("qsa_relu_order backend=%s blocks=%lld heads=%lld tokens=%lld streams=%lld keep=%d dense=%d first=%d reuse=%d reserve_bytes=%zu phases=2 PASSED\n",
                ggml_backend_name(backend), (long long) shape.blocks, (long long) shape.heads,
                (long long) shape.tokens, (long long) shape.streams, int(shape.keep), shape.dense, relu_first, reuse, reserved);
    }
    return result;
}

static size_t qsa_relu_symbolic_reserve(bool relu_first) {
    const qsa_relu_shape shape{ 52544, 4, 2048, 1 };
    auto reserve = make_qsa_relu_graph(shape, relu_first);
    dummy_backend backend = dummy_backend_init(SIZE_MAX, 64);
    ggml_gallocr_ptr galloc(ggml_gallocr_new(&backend.buffer_type));
    size_t size = 0;
    ggml_gallocr_reserve_n_size(galloc.get(), reserve.memory.graph, nullptr, nullptr, &size);
    GGML_ASSERT(size > ggml_nbytes(reserve.product));
    GGML_ASSERT(backend.context->allocated_total() == 0);
    for (int i = 0; i < ggml_graph_n_nodes(reserve.memory.graph); ++i) {
        GGML_ASSERT(ggml_graph_node(reserve.memory.graph, i)->data == nullptr);
    }
    return size;
}

static int test_qsa_relu_order(const char * backend_name, bool compare_orders = false) {
    GGML_ASSERT(!compare_orders || strcmp(backend_name, "Vulkan0") == 0);
    ggml_backend_load_all();
    ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
    ggml_backend_ptr other;
    GGML_ASSERT(cpu);
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    ggml_backend_t backend = cpu.get();
    if (strcmp(backend_name, "CPU") != 0) {
        other.reset(ggml_backend_init_by_name(backend_name, nullptr));
        GGML_ASSERT(other);
        backend = other.get();
    }
    GGML_ASSERT(!compare_orders || strcmp(ggml_backend_name(backend), "Vulkan0") == 0);
    const qsa_relu_shape cases[] = {
        { 17, 1, 1, 1 }, { 17, 4, 1, 1 }, { 17, 4, 3, 1 }, { 17, 4, 8, 1 },
        { 17, 4, 9, 1 }, { 17, 4, 65, 1 }, { 19, 4, 7, 2 },
        { 19, 4, 9, 2, QSA_RELU_KEEP_CHILD },
        { 19, 4, 9, 2, QSA_RELU_KEEP_VIEW },
        { 19, 4, 9, 2, QSA_RELU_KEEP_OUTPUT },
        { 257, 4, 65, 1, QSA_RELU_KEEP_NONE, true },
    };
    for (const auto & shape : cases) {
        const auto old_order = compute_qsa_relu_order(backend, cpu.get(), shape, false, compare_orders);
        const auto new_order = compute_qsa_relu_order(backend, cpu.get(), shape, true, compare_orders);
        GGML_ASSERT(old_order == new_order);
        if (compare_orders) {
            printf("qsa_relu_compare_pair backend=%s blocks=%lld heads=%lld tokens=%lld streams=%lld keep=%d dense=%d bytes=%zu phases=2 reorder_bytes_equal=true\n",
                    ggml_backend_name(backend), (long long) shape.blocks, (long long) shape.heads,
                    (long long) shape.tokens, (long long) shape.streams, int(shape.keep), shape.dense, old_order.size());
        }
    }
    const size_t old_size = qsa_relu_symbolic_reserve(false);
    const size_t new_size = qsa_relu_symbolic_reserve(true);
    GGML_ASSERT(new_size < old_size);
    if (compare_orders) {
        printf("qsa_relu_compare_symbolic blocks=52544 heads=4 tokens=2048 old_bytes=%zu new_bytes=%zu allocated_bytes=0 COMPLETE\n",
                old_size, new_size);
    } else {
        printf("qsa_relu_symbolic blocks=52544 heads=4 tokens=2048 old_bytes=%zu new_bytes=%zu allocated_bytes=0 PASSED\n",
                old_size, new_size);
    }
    return 0;
}

static void run(const char * name, void (*f)()) {
    printf("%s ", name);
    fflush(stdout);
    f();
    printf("PASSED\n");
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "--qsa-relu-order-compare") == 0) {
        return test_qsa_relu_order(argv[2], true);
    }
    if (argc == 3 && strcmp(argv[1], "--qsa-relu-order") == 0) {
        return test_qsa_relu_order(argv[2]);
    }
    if ((argc == 2 || argc == 3) && strcmp(argv[1], "--vulkan-conv-input") == 0) {
        return test_vulkan_conv_input(argc == 3 ? argv[2] : nullptr);
    }
    if (argc == 2 && strcmp(argv[1], "--vulkan-qsa") == 0) {
        return test_vulkan_qsa_alloc();
    }
    GGML_ASSERT(argc == 1);
    run("test_max_size_too_many_tensors", test_max_size_too_many_tensors);
    run("test_max_size_tensor_too_large", test_max_size_tensor_too_large);
    run("test_tensor_larger_than_max_size", test_tensor_larger_than_max_size);
    run("test_not_enough_chunks", test_not_enough_chunks);
    run("test_fill_leftover_space", test_fill_leftover_space);
    run("test_view_inplace", test_view_inplace);
    run("test_reuse_and_free", test_reuse_and_free);
    run("test_merge_free_block(32)", []() { test_merge_free_block(32); });
    run("test_merge_free_block(SIZE_MAX)", []() { test_merge_free_block(SIZE_MAX); });
    run("test_prefer_already_allocated_memory", test_prefer_already_allocated_memory);
    run("test_multiple_buffer_types", test_multiple_buffer_types);
    run("test_buffer_size_zero", test_buffer_size_zero);
    run("test_reallocation", test_reallocation);
    run("test_graph_optimize_alloc_dep", test_graph_optimize_alloc_dep);
    return 0;
}
