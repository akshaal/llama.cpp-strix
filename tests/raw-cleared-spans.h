#pragma once
#include "json.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
static void spans_probe_require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static int spans_probe_integer(const common_json & object, const char * key, int low, int high) {
    const auto & value = object.at(key);
    spans_probe_require(value.is_number_integer(), "diagnostic integer field has wrong type");
    const int64_t integer = value.get<int64_t>();
    spans_probe_require(integer >= low && integer <= high, "diagnostic integer field is out of bounds");
    return int(integer);
}

static common_json spans_probe_metadata(llama_memory_hybrid_idx * memory, llama_memory_recurrent * recurrent) {
    spans_probe_require(recurrent->cells.size() == 1 && recurrent->rs_idx.size() == 1, "unexpected recurrent cell count");
    const auto & cell = recurrent->cells[0];
    const uint32_t pooled_rows = memory->get_pooled_rows();
    return {
        {"head", recurrent->head}, {"used", recurrent->used}, {"size", recurrent->size},
        {"n", recurrent->n}, {"rs_z", recurrent->rs_z}, {"rs_idx", recurrent->rs_idx[0]},
        {"cell_pos", cell.pos}, {"cell_src", cell.src}, {"cell_src0", cell.src0}, {"cell_tail", cell.tail},
        {"cell_seq_ids", std::vector<llama_seq_id>(cell.seq_id.begin(), cell.seq_id.end())},
        {"pos_min", memory->seq_pos_min(0)}, {"pos_max", memory->seq_pos_max(0)},
        {"pooled_rows", pooled_rows},
        {"pooled_valid", pooled_rows ? common_json(memory->pooled_valid(0)) : common_json(nullptr)},
    };
}

enum class spans_probe_publish_fault { none, write, file_sync, file_close, directory_sync };

static void spans_probe_write_file(const std::filesystem::path & path, const void * source, size_t size,
                            spans_probe_publish_fault fault = spans_probe_publish_fault::none) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    spans_probe_require(fd >= 0, "cannot create diagnostic file");
    try {
        const auto * data = static_cast<const uint8_t *>(source);
        size_t written = 0;
        while (written < size) {
            const size_t part = fault == spans_probe_publish_fault::write ? std::min(size_t(8), size - written) : size - written;
            const ssize_t count = write(fd, data + written, part);
            if (count < 0 && errno == EINTR) { continue; }
            spans_probe_require(count > 0, "diagnostic write failed");
            written += size_t(count);
            spans_probe_require(fault != spans_probe_publish_fault::write, "injected marker write failure");
        }
        spans_probe_require(fault != spans_probe_publish_fault::file_sync, "injected marker file sync failure");
        spans_probe_require(fsync(fd) == 0, "diagnostic fsync failed");
    } catch (...) {
        close(fd);
        throw;
    }
    spans_probe_require(close(fd) == 0, "diagnostic close failed");
    spans_probe_require(fault != spans_probe_publish_fault::file_close, "injected marker close failure");
}

static int raw_cleared_spans(common_params params, const char * input_path, const char * output_path) {
    using json = common_json;
    try {
        spans_probe_require(std::filesystem::file_size(input_path) <= 1024 * 1024, "input too large");
        std::ifstream stream(input_path);
        spans_probe_require(stream.is_open(), "cannot open job");
        const std::string job_text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        spans_probe_require(!stream.bad(), "job read failed");
        const json job = json::parse(job_text);
        spans_probe_require(job.at("protocol") == "raw-cleared-spans-v1", "wrong job protocol");
        const auto & spec = job.at("a");
        const auto & requested = spec.at("context");
        const int n_vocab = spans_probe_integer(spec, "n_vocab", 128, 248320);
        spans_probe_require(n_vocab == 128 || n_vocab == 248320, "unsupported vocabulary");
        const int n_ctx_actual = n_vocab == 128 ? 2048 : 210176;
        params.n_ctx = n_vocab == 128 ? 2048 : 210000;
        params.n_batch = params.n_ubatch = 2048;
        params.cpuparams.n_threads = params.cpuparams_batch.n_threads = 16;
        const json expected = {{"n_ctx", params.n_ctx}, {"n_ctx_actual", n_ctx_actual},
            {"n_batch", 2048}, {"n_ubatch", 2048}, {"n_threads", 16}, {"n_threads_batch", 16},
            {"flash_attn", true}, {"cache_type_k", "f16"}, {"cache_type_v", "f16"},
            {"kv_unified", false}, {"offload_kqv", true}, {"op_offload", true}};
        spans_probe_require(requested == expected, "context must match the saved diagnostic");
        spans_probe_require(params.lora_adapters.empty() && params.control_vectors.empty() &&
            !params.embedding && !params.cb_eval, "unsupported adapter or callback");
        std::vector<std::vector<llama_token>> inputs;
        for (const auto & list : {spec.at("tokens"), job.at("b")}) {
            spans_probe_require(list.is_array() && list.size() == 1036, "expected 1036 input tokens");
            std::vector<llama_token> tokens;
            for (const auto & value : list) {
                spans_probe_require(value.is_number_integer() && value.get<int64_t>() >= 0 &&
                    value.get<int64_t>() < n_vocab, "invalid token");
                tokens.push_back(value.get<llama_token>());
            }
            inputs.push_back(std::move(tokens));
        }
        spans_probe_require(inputs[0] != inputs[1], "foreign window must differ");
        const auto & quality = job.at("windows");
        spans_probe_require(quality.is_array() && quality.size() == size_t(n_vocab == 128 ? 2 : 32), "wrong quality window count");
        std::set<std::string> identifiers;
        for (const auto & window : quality) {
            spans_probe_require(window.at("prefix_tokens") == 1024 && window.at("score_tokens") == 128 &&
                identifiers.insert(window.at("id").get<std::string>()).second, "wrong or duplicate quality window");
            const auto & values = window.at("tokens");
            spans_probe_require(values.is_array() && values.size() == 1152, "quality history must contain 1152 tokens");
            std::vector<llama_token> tokens;
            for (const auto & value : values) {
                spans_probe_require(value.is_number_integer() && value.get<int64_t>() >= 0 &&
                    value.get<int64_t>() < n_vocab, "invalid quality token");
                tokens.push_back(value.get<llama_token>());
            }
            inputs.push_back(std::move(tokens));
        }
        const size_t checkpoint_bytes = spans_probe_integer(job, "checkpoint_bytes", 1, 128 * 1024 * 1024);
        spans_probe_require(checkpoint_bytes == (n_vocab == 128 ? 177212 : 118039428), "checkpoint size differs");
        const uint16_t endian = 1;
        spans_probe_require(*reinterpret_cast<const uint8_t *>(&endian) == 1 && sizeof(float) == 4 &&
            std::numeric_limits<float>::is_iec559, "requires little-endian IEEE F32");
        const size_t window_floats = 13 * size_t(n_vocab);
        const std::filesystem::path reference_path = job.at("reference").get<std::string>();
        spans_probe_require(std::filesystem::file_size(reference_path) == window_floats * 4, "wrong FIRST13 size");
        std::vector<float> reference(window_floats), logits(window_floats);
        std::ifstream reference_stream(reference_path, std::ios::binary);
        reference_stream.read(reinterpret_cast<char *>(reference.data()), window_floats * 4);
        spans_probe_require(bool(reference_stream), "FIRST13 read failed");
        for (float x : reference) { spans_probe_require(std::isfinite(x), "nonfinite reference"); }
        const std::filesystem::path output(output_path);
        spans_probe_require(std::filesystem::create_directory(output), "output must be new");
        params.fit_params = false;
        params.warmup = false;
        params.n_parallel = 1;
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = 2;
        params.sampling.backend_sampling = false;
        params.cache_type_k = params.cache_type_v = GGML_TYPE_F16;
        params.flash_attn_type = requested.at("flash_attn").get<bool>() ?
            LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        params.kv_unified = requested.at("kv_unified").get<bool>();
        params.no_kv_offload = !requested.at("offload_kqv").get<bool>();
        params.no_op_offload = !requested.at("op_offload").get<bool>();
        common_init_result_ptr initialized = common_init_from_params(params, true);
        llama_model * model = initialized->model();
        spans_probe_require(model != nullptr && initialized->context() == nullptr, "model-only initialization failed");
        char arch[64] = {};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        spans_probe_require(strcmp(arch, "qwen4exp") == 0 &&
                       llama_vocab_n_tokens(llama_model_get_vocab(model)) == n_vocab, "wrong diagnostic model");
        common_threadpools threadpools;
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max = 1;
        cparams.n_rs_seq = 2;
        cparams.n_outputs_max = cparams.n_outputs_max_per_seq = 3;
        cparams.samplers = nullptr;
        cparams.n_samplers = 0;
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        llama_context_ptr context(llama_init_from_model(model, cparams));
        spans_probe_require(context != nullptr, "diagnostic context initialization failed");
        llama_context * ctx = context.get();
        threadpools.init(ctx, params);
        llama_set_embeddings_nextn(ctx, true, false);
        const auto & actual = ctx->get_cparams();
        spans_probe_require(actual.n_ctx == uint32_t(n_ctx_actual) &&
                       actual.n_batch == uint32_t(params.n_batch) && actual.n_ubatch == uint32_t(params.n_ubatch) &&
                       actual.n_seq_max == 1 && actual.n_rs_seq == 2 && actual.n_outputs_max == 3 &&
                       actual.n_outputs_max_per_seq == 3 && actual.embeddings_nextn &&
                       !actual.embeddings_nextn_masked && !actual.embeddings &&
                       actual.flash_attn == requested.at("flash_attn").get<bool>() &&
                       actual.kv_unified == params.kv_unified && actual.offload_kqv == !params.no_kv_offload &&
                       actual.op_offload == !params.no_op_offload &&
                       actual.n_threads == params.cpuparams.n_threads &&
                       actual.n_threads_batch == params.cpuparams_batch.n_threads,
                       "diagnostic actual context differs from requested settings");
        auto * memory = static_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
        spans_probe_require(memory != nullptr, "diagnostic target has no hybrid memory");
        auto * recurrent = memory->get_mem_recr();
        spans_probe_require(recurrent && recurrent->size == 1 && recurrent->n_rs_seq == 2,
            "unexpected recurrent layout");
        const auto capacity = memory->get_pooled_rows();
        spans_probe_require(capacity > 0 && actual.fused_gdn_ar && actual.fused_gdn_ch && actual.fused_lid &&
            actual.fused_dsv4_hc_pre && actual.fused_dsv4_hc_comb && actual.fused_dsv4_hc_post,
            "pooled capacity or fusion configuration differs");
        std::vector<uint8_t> checkpoint(checkpoint_bytes);
        json windows = json::array();
        const int order[] = {0, 0, 1, 0};
        for (int w = 0; w < 4 + int(quality.size()); ++w) {
            const int input_id = w < 4 ? order[w] : w - 2;
            const int rows = w < 4 ? 13 : 128;
            if (w == 4) {
                std::vector<float>().swap(reference);
                logits.resize(128 * size_t(n_vocab));
            }
            const json before_reset = spans_probe_metadata(memory, recurrent);
            llama_synchronize(ctx);
            llama_memory_clear(memory, true);
            const json after_reset = spans_probe_metadata(memory, recurrent);
            spans_probe_require(after_reset.at("head") == 0 && after_reset.at("used") == 0 &&
                after_reset.at("rs_idx") == 0 && after_reset.at("pos_min") == -1 &&
                after_reset.at("pos_max") == -1 && after_reset.at("cell_seq_ids").empty() &&
                after_reset.at("pooled_valid") == 0 && memory->get_pooled_rows() == capacity,
                "reset did not empty the cache metadata");
            json events = json::array();
            for (int step = 0; step <= rows; ++step) {
                const int begin = step == 0 ? 0 : step == 1 ? 1020 : 1022 + step;
                const int count = step == 0 ? 1020 : step == 1 ? 4 : 1;
                json event = {{"step", step}, {"begin", begin}, {"count", count}};
                if (step > 0) {
                    const json before = spans_probe_metadata(memory, recurrent);
                    spans_probe_require(memory->seq_pos_max(0) == begin - 1 && recurrent->rs_idx[0] == 0,
                        "wrong checkpoint position");
                    spans_probe_require(llama_memory_seq_rm(memory, 0, begin, -1), "empty removal failed");
                    spans_probe_require(before == spans_probe_metadata(memory, recurrent), "removal changed state");
                    const size_t size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    spans_probe_require(size == checkpoint_bytes, "checkpoint size changed");
                    spans_probe_require(llama_state_seq_get_data_ext(ctx, checkpoint.data(), size, 0,
                        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size, "checkpoint save failed");
                    spans_probe_require(before == spans_probe_metadata(memory, recurrent), "save changed state");
                    event["checkpoint"] = {{"bytes", size}, {"state", before}};
                }
                llama_batch batch = llama_batch_init(count, 0, 1);
                for (int i = 0; i < count; ++i) {
                    common_batch_add(batch, inputs[input_id][begin + i], begin + i, {0}, step > 0 && i == count - 1);
                }
                const int code = llama_decode(ctx, batch);
                llama_batch_free(batch);
                spans_probe_require(code == 0, "decode failed");
                llama_synchronize(ctx);
                spans_probe_require(memory->seq_pos_max(0) == begin + count - 1 &&
                    memory->seq_pos_min(0) == begin + count - 1 && recurrent->used == 1 &&
                    recurrent->rs_idx[0] == 0 && memory->get_pooled_rows() == capacity,
                    "post-decode state differs");
                event["after"] = spans_probe_metadata(memory, recurrent);
                if (step > 0) {
                    const int index = count - 1;
                    spans_probe_require(llama_get_sampled_token_ith(ctx, index) == LLAMA_TOKEN_NULL &&
                        !llama_get_sampled_logits_ith(ctx, index) && !llama_get_sampled_probs_ith(ctx, index),
                        "unexpected backend sampler");
                    const float * row = llama_get_logits_ith(ctx, index);
                    spans_probe_require(row != nullptr, "missing raw logits");
                    for (int id = 0; id < n_vocab; ++id) { spans_probe_require(std::isfinite(row[id]), "nonfinite logits"); }
                    memcpy(logits.data() + (step - 1) * size_t(n_vocab), row, n_vocab * 4);
                    event["position"] = begin + count;
                    event["output_index"] = index;
                    if (w >= 4) { event["gold"] = inputs[input_id][begin + count]; }
                }
                events.push_back(std::move(event));
            }
            const float * current = logits.data();
            spans_probe_write_file(output / (std::to_string(w) + ".f32"), current, rows * size_t(n_vocab) * 4);
            windows.push_back({{"window", w}, {"input", input_id}, {"before_reset", before_reset},
                {"after_reset", after_reset}, {"events", events}});
            const std::string record = windows.back().dump(2) + "\n";
            spans_probe_write_file(output / (std::to_string(w) + ".json"), record.data(), record.size());
            if (w < 4 && order[w] == 0) {
                spans_probe_require(memcmp(current, reference.data(), window_floats * 4) == 0,
                    "saved original FIRST13 byte gate failed");
            }
            fprintf(stderr, "cleared-spans: window %d complete\n", w);
        }
        const std::string result = json({{"protocol", "raw-cleared-spans-result-v1"}, {"job", job},
            {"windows", windows}, {"first13_and_aa_b_a_exact", true},
            {"quality_capture_complete", n_vocab == 248320}, {"smoke_complete", n_vocab == 128}}).dump(2) + "\n";
        spans_probe_write_file(output / "result.json", result.data(), result.size());
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "cleared-spans failed: %s\n", error.what());
        return 1;
    }
}
