#pragma once

// A paired state probe, not a sampler or an end-to-end quality benchmark.
#include "speculative.h"
#include "../src/llama-batch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace mtp_cache_probe {

static constexpr const char * flag = "LLAMA_QWEN4EXP_MTP_CACHE_ONLY";
static constexpr size_t image_limit = 64 * 1024 * 1024;
static constexpr size_t reference_limit = 256 * 1024 * 1024;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void append(std::vector<uint8_t> & dst, const void * data, size_t size) {
    require(size <= image_limit && dst.size() <= image_limit - size, "state image exceeds bound");
    if (size) {
        const auto * first = static_cast<const uint8_t *>(data);
        dst.insert(dst.end(), first, first + size);
    }
}

template<class T> static void scalar(std::vector<uint8_t> & dst, T value) {
    append(dst, &value, sizeof(value));
}

struct writer : llama_io_write_i {
    std::vector<uint8_t> logical;
    std::vector<ggml_tensor *> tensors;

    void write(const void * data, size_t size) override { append(logical, data, size); }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        require(offset <= ggml_nbytes(tensor) && size <= ggml_nbytes(tensor) - offset,
                "serialized tensor range is invalid");
        require(size <= image_limit && logical.size() <= image_limit - size, "logical state exceeds bound");
        const size_t start = logical.size();
        logical.resize(start + size);
        ggml_backend_tensor_get(tensor, logical.data() + start, offset, size);
        if (std::find(tensors.begin(), tensors.end(), tensor) == tensors.end()) { tensors.push_back(tensor); }
    }
    size_t n_bytes() override { return logical.size(); }
};

struct image {
    std::string label;
    std::vector<uint8_t> logical, physical, cells;
    std::vector<float> hidden, logits;
    size_t size() const {
        return logical.size() + physical.size() + cells.size() + 4 * (hidden.size() + logits.size());
    }
};

template<class T> static void exact(const std::vector<T> & a, const std::vector<T> & b,
                                   const char * label, const char * part) {
    require(a.size() == b.size(), "paired image lengths differ");
    const size_t size = a.size() * sizeof(T);
    if (size && std::memcmp(a.data(), b.data(), size) != 0) {
        const auto * x = reinterpret_cast<const uint8_t *>(a.data());
        const auto * y = reinterpret_cast<const uint8_t *>(b.data());
        size_t first = 0, changed = 0;
        while (first < size && x[first] == y[first]) { ++first; }
        for (size_t i = first; i < size; ++i) { changed += x[i] != y[i]; }
        fprintf(stderr, "MTP_CACHE_ONLY_DIFF step=%s part=%s first_byte=%zu changed_bytes=%zu size=%zu\n",
                label, part, first, changed, size);
        throw std::runtime_error("paired bytes differ");
    }
}

static image capture(llama_context * ctx, const char * label, int past, bool live, int width, int vocab) {
    llama_synchronize(ctx);
    auto * memory = dynamic_cast<llama_kv_cache *>(llama_get_memory(ctx));
    require(memory != nullptr, "draft memory is not an ordinary KV cache");
    writer state;
    memory->state_write(state);
    require(state.tensors.size() == 2, "expected exactly one draft K tensor and one V tensor");
    image result;
    result.label = label;
    result.logical = std::move(state.logical);
    for (ggml_tensor * tensor : state.tensors) {
        require(tensor->type == GGML_TYPE_F16 && ggml_is_contiguous(tensor), "probe requires packed F16 cache tensors");
        const size_t size = ggml_nbytes(tensor);
        require(size > 0 && size <= image_limit / 2 && size % sizeof(ggml_fp16_t) == 0,
                "invalid full cache tensor size");
        scalar(result.physical, int32_t(tensor->type));
        for (int i = 0; i < 4; ++i) {
            scalar(result.physical, int64_t(tensor->ne[i]));
            scalar(result.physical, uint64_t(tensor->nb[i]));
        }
        std::vector<ggml_fp16_t> data(size / sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(tensor, data.data(), 0, size);
        bool nonzero = false;
        for (ggml_fp16_t value : data) {
            const float f = ggml_fp16_to_fp32(value);
            require(std::isfinite(f), "nonfinite cache value");
            nonzero |= f != 0.0f;
        }
        require(nonzero, "cache payload is vacuously all zero");
        append(result.physical, data.data(), size); // Includes unused and rejected physical rows.
    }

    const auto & cells = memory->get_cells(0);
    require(past > 0 && cells.get_used() == uint32_t(past), "wrong occupied-cell count");
    std::vector<bool> seen(past, false);
    scalar(result.cells, cells.size());
    for (uint32_t i = 0; i < cells.size(); ++i) {
        const bool empty = cells.is_empty(i);
        const bool seq0 = cells.seq_has(i, 0);
        const llama_pos pos = empty ? -1 : cells.pos_get(i);
        scalar(result.cells, uint8_t(empty));
        scalar(result.cells, uint8_t(seq0));
        scalar(result.cells, pos);
        // The getters reject empty cells. Their old ext bytes have no live metadata contract.
        const llama_kv_cell_ext ext = empty ? llama_kv_cell_ext{} : cells.ext_get(i);
        scalar(result.cells, ext.x);
        scalar(result.cells, ext.y);
        scalar(result.cells, ext.tok);
        const llama_pos shift = empty ? 0 : cells.get_shift(i);
        scalar(result.cells, shift);
        require(shift == 0, "unexpected shifted cell");
        require(empty == !seq0, "unexpected sequence ownership");
        if (!empty) {
            require(pos >= 0 && pos < past && !seen[pos], "missing or duplicate logical position");
            seen[pos] = true;
        }
    }
    require(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }), "logical prefix is incomplete");
    require(llama_memory_seq_pos_min(memory, 0) == 0 && llama_memory_seq_pos_max(memory, 0) == past - 1,
            "wrong draft position bounds");
    if (live) {
        const float * hidden = llama_get_embeddings_nextn_ith(ctx, 0);
        const float * logits = llama_get_logits_ith(ctx, 0);
        require(hidden && logits, "missing live output");
        result.hidden.assign(hidden, hidden + width);
        result.logits.assign(logits, logits + vocab);
        for (const auto * values : { &result.hidden, &result.logits }) {
            bool nonzero = false;
            for (float value : *values) {
                require(std::isfinite(value), "nonfinite live output");
                nonzero |= value != 0.0f;
            }
            require(nonzero, "live output is vacuously all zero");
        }
    }
    return result;
}

static void topology_contract(llama_context * ctx, const llama_batch & inputs, int tokens, int outputs, int expected_fa) {
    require(tokens > 0 && (outputs == 0 || (tokens == 1 && outputs == 1)) && inputs.token && inputs.embd,
            "invalid topology probe inputs");
    const auto & model = ctx->get_model();
    auto memory = ctx->get_memory()->init_full();
    require(memory != nullptr, "missing topology probe memory context");
    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    auto ubatch = balloc.ubatch_reserve(tokens, 1);
    require(ubatch.n_seqs == 1 && ubatch.n_seqs_unq == 1 && ubatch.seq_id_unq[0] == 0,
            "unexpected topology probe sequence");
    ubatch.token = inputs.token;
    ubatch.embd = inputs.embd;
    for (int i = 0; i < tokens; ++i) {
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &ubatch.seq_id_unq[0];
        ubatch.output[i] = outputs != 0;
    }
    llama_adapter_cvec cvec;
    llama_adapter_loras loras;
    llama_cross cross;
    llm_graph_result result(ctx->graph_max_nodes(tokens));
    llm_graph_params params = {};
    params.arch = model.arch;
    params.hparams = model.hparams;
    params.cparams = ctx->get_cparams();
    params.ubatch = ubatch;
    params.gtype = LLM_GRAPH_TYPE_DECODER_MTP;
    params.cvec = &cvec;
    params.loras = &loras;
    params.mctx = memory.get();
    params.cross = &cross;
    params.prec_policy = &model.prec_policy;
    params.n_outputs = outputs;
    params.res = &result;
    require(params.cparams.flash_attn, "topology probe requires flash attention");
    // Build metadata only. Do not reserve, set inputs, compute or change the context's graph cache.
    ggml_cgraph * graph = model.build_graph(params);
    require(graph != nullptr, "topology graph construction failed");
    int fa = 0, stores = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        fa += node->op == GGML_OP_FLASH_ATTN_EXT;
        stores += node->op == GGML_OP_SET_ROWS;
    }
    require(fa == expected_fa && stores == 2, "topology graph did not select the expected FA/KV path");
    fprintf(stdout, "MTP_CACHE_ONLY_GRAPH source=metadata tokens=%d outputs=%d fa=%d kv_stores=%d\n", tokens, outputs, fa, stores);
}

struct batch_owner {
    llama_batch batch;
    batch_owner(int count, int width) : batch(llama_batch_init(count, width, 1)) {
        batch.token = static_cast<llama_token *>(std::malloc(sizeof(llama_token) * count));
        if (!batch.token) { llama_batch_free(batch); throw std::bad_alloc(); }
    }
    ~batch_owner() { llama_batch_free(batch); }
};

static std::vector<image> arm(common_params params, llama_model * model, llama_context * target,
                              int prefix, int width, int vocab, bool enabled,
                              const std::vector<image> * reference) {
    require(setenv(flag, enabled ? "1" : "0", 1) == 0, "cannot set candidate flag");
    auto cp = common_context_params_to_llama(params);
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.ctx_other = target;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 0;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 1;
    cp.samplers = nullptr;
    cp.n_samplers = 0;
    cp.cb_eval = nullptr; // No tensor observer or its extra per-split synchronization.
    cp.cb_eval_user_data = nullptr;
    common_threadpools threadpools;
    llama_context_ptr context(llama_init_from_model(model, cp));
    require(context != nullptr, "draft context creation failed");
    llama_context * ctx = context.get();
    threadpools.init(ctx, params);
    llama_set_embeddings_nextn(ctx, true, true);
    require(llama_n_ctx(ctx) >= uint32_t(prefix + 16) && llama_n_ctx(ctx) <= 4096 &&
            llama_n_ubatch(ctx) >= uint32_t(prefix), "actual draft context is outside probe bounds");

    batch_owner owned(prefix, width);
    auto & batch = owned.batch;
    topology_contract(ctx, batch, prefix, 0, enabled ? 0 : 1);
    topology_contract(ctx, batch, 1, 1, 1);
    llama_set_embeddings_nextn(ctx, false, true);
    topology_contract(ctx, batch, 3, 0, 1); // A zero-output fallback when hidden export is disabled.
    llama_set_embeddings_nextn(ctx, true, true);
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), true);

    std::vector<image> saved;
    std::vector<float> draft_hidden;
    std::vector<uint8_t> previous_physical;
    llama_token draft_token = 0;
    size_t total = 0, step = 0;
    int zero_decodes = 0, live_decodes = 0;
    const auto emit = [&](const char * label, int past, bool live, bool removal) {
        image value = capture(ctx, label, past, live, width, vocab);
        if (removal) { exact(previous_physical, value.physical, label, "seq_rm_payload_preservation"); }
        previous_physical = value.physical;
        if (live) {
            draft_hidden = value.hidden;
            draft_token = llama_token(std::max_element(value.logits.begin(), value.logits.end()) - value.logits.begin());
        }
        if (reference) {
            require(step < reference->size() && (*reference)[step].label == label, "paired schedule differs");
            const auto & old = (*reference)[step];
            exact(old.logical, value.logical, label, "serialized_state");
            exact(old.physical, value.physical, label, "full_KV");
            exact(old.cells, value.cells, label, "physical_cells");
            exact(old.hidden, value.hidden, label, "next_hidden");
            exact(old.logits, value.logits, label, "next_logits");
        } else {
            require(value.size() <= reference_limit - total, "reference capture exceeds bound");
            total += value.size();
            saved.push_back(std::move(value));
        }
        fprintf(stdout, "MTP_CACHE_ONLY_STATE arm=%d step=%zu label=%s past=%d live=%d exact=%d\n",
                int(enabled), step, label, past, int(live), reference != nullptr);
        ++step;
    };
    const auto decode = [&](const char * label, int pos, int count, bool live, bool feedback, int variant) {
        require(count > 0 && count <= prefix && (!feedback || (count == 1 && int(draft_hidden.size()) == width)),
                "invalid probe decode");
        common_batch_clear(batch);
        for (int i = 0; i < count; ++i) {
            const int p = pos + i;
            const llama_token token = feedback ? draft_token : 3 + (17 * p + variant) % (vocab - 3);
            common_batch_add(batch, token, p, { 0 }, live);
            for (int j = 0; j < width; ++j) {
                // Fixed, nonzero synthetic target hidden rows. The first pending row is zero.
                batch.embd[size_t(i) * width + j] = feedback ? draft_hidden[j] :
                        p == 0 ? 0.0f : float((13 * (p - 1) + 7 * j + variant) % 257 - 128) / 256.0f;
            }
        }
        require(llama_decode(ctx, batch) == 0, "draft decode failed");
        live ? ++live_decodes : ++zero_decodes;
        emit(label, pos + count, live, false);
    };

    decode("prompt", 0, prefix, false, false, 1);
    decode("anchor", prefix, 1, true, false, 1);
    decode("draft-one", prefix + 1, 1, true, true, 1);
    decode("draft-two", prefix + 2, 1, true, true, 1);
    require(llama_memory_seq_rm(llama_get_memory(ctx), 0, prefix + 2, -1), "suffix removal failed");
    emit("accept-one-reject-two", prefix + 2, false, true);
    decode("replacement-catchup", prefix + 2, 1, false, false, 97);
    decode("after-replacement", prefix + 3, 1, true, false, 97);
    decode("accepted-catchup-three", prefix + 4, 3, false, false, 53);
    const int reused_before = llama_perf_context(ctx).n_reused;
    decode("reused-catchup-three", prefix + 7, 3, false, false, 89);
    require(llama_perf_context(ctx).n_reused > reused_before, "same-shape zero-output graph was not reused");
    decode("after-reused-three", prefix + 10, 1, true, false, 89);
    decode("second-live-after-three", prefix + 11, 1, true, false, 89);
    llama_set_embeddings_nextn(ctx, false, true);
    decode("exports-off-fallback", prefix + 12, 3, false, false, 131);
    llama_set_embeddings_nextn(ctx, true, true);
    decode("after-fallback", prefix + 15, 1, true, false, 131);
    require(step == 13 && zero_decodes == 5 && live_decodes == 7 &&
            (!reference || step == reference->size()), "incomplete paired schedule");
    fprintf(stdout, "MTP_CACHE_ONLY_ARM arm=%d snapshots=%zu zero=%d live=%d graph_reuse=1 reference_bytes=%zu\n",
            int(enabled), step, zero_decodes, live_decodes, total);
    return saved;
}

static int run(common_params params) {
    try {
        require(std::getenv(flag) == nullptr && std::getenv("LLAMA_GRAPH_REUSE_DISABLE") == nullptr,
                "unset candidate and graph-reuse overrides before running the paired probe");
        struct reset_env { ~reset_env() { unsetenv(flag); } } restore;
        require(params.n_ctx > 0 && params.n_ctx <= 4096 && params.lora_adapters.empty() &&
                params.control_vectors.empty(), "probe requires context <=4096 and no adapters/control vectors");
        params.fit_params = false;
        params.warmup = false;
        params.n_parallel = 1;
        params.no_perf = false;
        params.embedding = false;
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = 2;
        // The COMMON test parser does not expose the server's -md option.
        if (const char * path = std::getenv("LLAMA_TEST_MTP_DRAFT_MODEL")) {
            require(*path != '\0', "empty draft model path");
            params.speculative.draft.mparams.path = path;
        }
        params.cache_type_k = params.cache_type_v = GGML_TYPE_F16;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.sampling.backend_sampling = false;
        auto target_model = common_init_from_params(params, true);
        require(target_model && target_model->model(), "target model load failed");
        llama_model * target_weights = target_model->model();
        char arch[64] = {};
        llama_model_meta_val_str(target_weights, "general.architecture", arch, sizeof(arch));
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(target_weights));
        const int width = llama_model_n_embd_out(target_weights);
        require(std::strcmp(arch, "qwen4exp") == 0 && (vocab == 128 || vocab == 248320), "wrong probe model");
        const int prefix = vocab == 128 ? 64 : 2048;
        require(params.n_ctx >= prefix + 16 && params.n_batch >= prefix && params.n_ubatch >= prefix &&
                width > 0 && width <= 10240, "probe context/batch/hidden width is outside bounds");
        auto target_cp = common_context_params_to_llama(params);
        target_cp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        target_cp.n_seq_max = 1;
        target_cp.n_rs_seq = 2;
        target_cp.n_outputs_max = target_cp.n_outputs_max_per_seq = 1;
        target_cp.samplers = nullptr;
        target_cp.n_samplers = 0;
        llama_context_ptr target(llama_init_from_model(target_weights, target_cp));
        require(target != nullptr, "target context creation failed");
        auto draft_params = common_base_params_to_speculative(params);
        common_init_result_ptr draft_model;
        llama_model * draft_weights = target_weights;
        if (params.speculative.has_dft()) {
            draft_model = common_init_from_params(draft_params, true);
            require(draft_model && draft_model->model(), "shared draft model load failed");
            draft_weights = draft_model->model();
        }
        require(llama_model_n_layer_nextn(draft_weights) == 1 && llama_model_n_embd_out(draft_weights) == width,
                "probe requires a real single-block MTP model with the target hidden width");
        const auto reference = arm(draft_params, draft_weights, target.get(), prefix, width, vocab, false, nullptr);
        arm(draft_params, draft_weights, target.get(), prefix, width, vocab, true, &reference);
        fprintf(stdout, "MTP_CACHE_ONLY_PROBE PASS vocab=%d prefix=%d snapshots=13 live_rows=7 full_KV=exact metadata=exact hidden=exact logits=exact\n",
                vocab, prefix);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "MTP_CACHE_ONLY_PROBE FAIL: %s\n", e.what());
        return 1;
    }
}

} // namespace mtp_cache_probe
