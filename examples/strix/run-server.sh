#!/usr/bin/env bash
set -euo pipefail
export LLAMA_QWEN4EXP_MTP_CACHE_ONLY=1
export GGML_VK_MMQ_ID_COMPACT=1

STRIX_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${STRIX_MODEL:?Set STRIX_MODEL to the first UD-Q4_K_XL shard.}"
STRIX_SERVER="${STRIX_SERVER:-$STRIX_SOURCE_DIR/build-strix/bin/llama-server}"
STRIX_DRAFT="${STRIX_DRAFT:-}"

[[ -x "$STRIX_SERVER" ]] || { printf 'Server binary not found: %s\n' "$STRIX_SERVER" >&2; exit 1; }
[[ -r "$STRIX_MODEL" ]] || { printf 'Model shard not found: %s\n' "$STRIX_MODEL" >&2; exit 1; }

STRIX_COMMAND=(
    "$STRIX_SERVER" -m "$STRIX_MODEL"
    --device Vulkan0 --n-gpu-layers all
    --load-mode mmap --lazy-mode on --fit off
    --ctx-size 210000 --parallel 1
    --cache-type-k f16 --cache-type-v f16 -fa on
    --batch-size 2048 --ubatch-size 2048 --threads 16
    --temp 1 --top-p 0.95 --top-k 20 --min-p 0.00
    --presence-penalty 0 --repeat-penalty 1.0
    --chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"xhigh"}'
    --no-mmproj --metrics --port 7000 --host 0.0.0.0
)

# Set STRIX_MTP=0 to use the same build without the draft head.
case "${STRIX_MTP:-1}" in
    1)
        [[ -r "$STRIX_DRAFT" ]] || { printf 'MTP head not found: %s\n' "$STRIX_DRAFT" >&2; exit 1; }
        STRIX_COMMAND+=(
            --spec-type draft-mtp --spec-draft-model "$STRIX_DRAFT"
            --spec-draft-n-max 2 --spec-draft-n-min 1 --spec-draft-p-min 0
            --spec-draft-ngl all --spec-draft-type-k f16 --spec-draft-type-v f16
        )
        ;;
    0) ;;
    *) printf 'STRIX_MTP must be 0 or 1.\n' >&2; exit 2 ;;
esac

exec "${STRIX_COMMAND[@]}" "$@"
