# What the commit sequence contains

This history was reconstructed from the frozen checkpoint patches. The order makes dependencies visible. It is not a record of the original editing sessions, and the commits are not backdated. Codex assembled the history with AI assistance disclosed in every new commit. Original upstream commits retain their authorship.

Start at the [tested upstream base](https://github.com/ggml-org/llama.cpp/commit/2145525a4081d66ff1a87cf43ef809f95a85ac0c), then read the added commits in order:

```sh
git log --reverse --format='%s' strix-upstream-base..strix-checkpoint-004
```

## Checkpoint 001 integrates the initial improvements

1. **Shared-head MTP runtime support.** Adapt [PR 28243](https://github.com/ggml-org/llama.cpp/pull/28243) to the tested base. The draft head borrows the target's embedding and output weights. The target still verifies proposed tokens.
2. **Buffered lazy PLE row reads.** Adapt [PR 29030](https://github.com/ggml-org/llama.cpp/pull/29030). Sort and deduplicate reads of the large embedding table while retaining the original quantized data.
3. **F16 KV repacking for AMD.** Adapt [Nathanw1014's layout change](https://github.com/Nathanw1014/llama.cpp/commit/ab5910a15e85b919b228193ed297a35beaf135c6). The cache stays F16. Guards restrict the new layout to supported shapes.
4. **F16 repacking tests.** Preserve the corresponding CPU-reference and fallback cases.
5. **Pooled QSA keys and layout fixes.** Integrate [PR 28699](https://github.com/ggml-org/llama.cpp/pull/28699), which caches pooled indexer keys. Local fixes cover rollback, invalidation, restore, clearing, accounting and fallback behavior. This saved combined patch also includes [Nathanw1014's pooling-copy removal](https://github.com/Nathanw1014/llama.cpp/commit/f347860db68eb0b493f29ec6568e8b147d165256), [apepojken's contiguous GDN input](https://github.com/apepojken/llama.cpp/commit/51c0c10d5904550dc822925f74367aa47982249c), and [wdenejko's deterministic top-k ordering](https://github.com/wdenejko/llama.cpp/commit/1114ac366fe284eca8129f059ee14cd43188522a). These dependent changes remain together in their saved patch.
6. **Pooled QSA rollback tests.** Exercise state restore and rejected-token replacement.
7. **Activation assertions.** Require the tests to use the optimized path. These checks caught an ordinary-text guard that was disabling it.
8. **Text and M-RoPE fix.** Correct that guard and position broadcasting for MTP batches.
9. **Sparse prefill dispatch.** Enable the existing Vulkan sparse attention kernel from [PR 28105](https://github.com/ggml-org/llama.cpp/pull/28105) for supported AMD prompt batches. This host adaptation was informed by [Nathanw1014's related experiments](https://github.com/Nathanw1014/llama.cpp/commit/002083fdb0dc143c98a516bd715917636965f70a). It retains the current kernel and conservative fallbacks.
10. **Sparse prefill tests.** Preserve the CPU-reference, activation and long-KV cases.
11. **8k crossover.** Use sparse prefill from the measured crossover onward. Smaller contexts keep the previous path.

## Checkpoint 002 removes memory copies

12. **Read QSA scores in place.** Coalesced selection reads the existing score layout and removes transpose/gather copies. Score arithmetic and selection stay the same. The saved declaration correction is included in this commit.
13. **Copy convolution inputs directly.** Copy new input and history columns into their final buffer with the existing integer-copy kernels. This removes intermediate memory traffic and preserves the computation.
14. **Copy and diagnostic tests.** Preserve the frozen allocator, replay and full-logit diagnostic sources. Some retained test cases describe experimental paths that are not enabled in the production source. Their inclusion is part of reproducing the exact checkpoint tree.

These are local implementations. Their combined measured benefit was 20.3% faster long prompt processing than 001. That combined measurement does not assign a separate whole-model gain to each patch.

## Checkpoint 003 avoids empty expert workgroups

15. **Compact expert dispatch and tests.** Add tile prefixes to the expert count pass and map a compact workgroup grid to occupied expert tiles. Multiplication and output arithmetic stay unchanged. Initialize padded row IDs and retain fallbacks. The feature is enabled with `GGML_VK_MMQ_ID_COMPACT=1`.

This is a local implementation of a known class of sparse dispatch optimization. No global novelty claim is made. Short prompt throughput improved 4.7% in the saved comparison. Its 0.7% long-prompt difference was too small to establish a repeatable gain.

## Checkpoint 004 skips unused MTP work

16. **Cache-only MTP catch-up.** During prompt catch-up the draft needs to populate K/V, but no output rows are requested. Preserve hidden mixing, K/V projections, normalization, RoPE, rotations and stores. Omit query projection, attention and the unused output tail. Live draft predictions keep their previous graph and arithmetic.
17. **Actual sampler and layout support.** Allow the initialized, stateless built-in top-k chain. Check the number of unique sequences because the real single-stream batch can have one sequence entry per token. Unsupported sampler or hidden-output modes keep the full graph.
18. **State and activation probes.** Preserve the paired cache probes, sampler checks and NEXTN fixture. Check full bytes and actual activation across reuse, rejection and replacement. Raw top-k arrays are unordered and are compared without sorting.

The cache-only path is enabled with `LLAMA_QWEN4EXP_MTP_CACHE_ONLY=1`. Its settings must remain fixed for the context lifetime. The basic KV-only idea also appears in [itsuwari/llama.cpp-qwen](https://github.com/itsuwari/llama.cpp-qwen/blob/807c805b2afe26f107fcd00d442f7ffbce875b23/src/models/qwen4exp.cpp#L607). This counterpart was identified after the local implementation. This fork makes no claim of being first.

Checkpoint 004 adds a measured 12.64% long-prompt throughput improvement over 003. See the [results and limits](README.md#measured-performance) before interpreting that figure.

## The last commit adds the fork documentation

19. **Fork guide, results and launcher.** Add the root README notice, these explanations, a compact validation summary and a launcher that accepts explicit model paths. Inference source remains identical to checkpoint 004.

The rejected fused SSM kernel, its host-scanning derivative, the alternative expert tiles, Gufo experiments and precision-substitution experiments are not part of the production implementation in this branch.
