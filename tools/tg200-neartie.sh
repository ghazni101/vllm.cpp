#!/bin/sh
# GFX1100-TG200 — teacher-forced near-tie adjudication wrapper.
#
# Every reduction-order lever owes the logprob-band ceremony per
# .agents/specs/rocm-m4-oracle.md (band <= 500 mnats) BEFORE it can claim its
# 256-token divergence is a near-tie. This wrapper runs the adjudicator binary
# (examples/tg200_neartie, built at build-hip-docker/examples/tg200-neartie)
# in the campaign container with the campaign reference config, under the
# gpu-ctl lock (the binary itself is GPU-free; the ENGINE is not).
#
# Usage:
#   tools/tg200-neartie.sh <capture|adjudicate> [KEY=VALUE ...] -- [binary args...]
#
#   KEY=VALUE tokens are exported INSIDE the container (the VT_* levers).
#   The literal token `@levers` expands to the campaign's adopted-lever block
#   (the T33/T34 reference config the 783cea17... reference body was captured
#   under). Everything after `--` is passed to the binary verbatim.
#
# Examples:
#   # reference capture (binds hard to the campaign reference body md5):
#   tools/tg200-neartie.sh capture @levers -- \
#     --model /models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf \
#     --prompt-file /repo/tg200/tools/tg200-prompt.txt \
#     --out /repo/tg200/tools/tg200-reference \
#     --max-tokens 256 --expect-md5 783cea1790ae7ebc4a0105fd309a6712
#
#   # adjudicate a lever arm against the committed reference:
#   tools/tg200-neartie.sh adjudicate @levers VT_MY_LEVER=1 -- \
#     --model /models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf \
#     --prompt-file /repo/tg200/tools/tg200-prompt.txt \
#     --ref-ids /repo/tg200/tools/tg200-reference.ids.i32 \
#     --json /job/my-lever-neartie.json --note "T40 my lever ON"
#
# Exit: the binary's verdict codes (0 PASS / 1 FAIL / 3 runtime / 4 integrity).
set -eu
cd /home/ghazni/github/vllm.cpp/tg200

MODE="$1"; shift

# ONE LINE per env list: the inner `for kv in $ENVS` splits on whitespace, and
# a raw newline inside a for-in word list is a syntax error, not a separator.
LEVERS="VT_GEMV_MMVQ=1 VT_SKINNY_BF16=1 VT_ATTN_DECODE_GQA4=1 VT_GDN_SCAN_COOP=1 VT_ATTN_PREAMBLE_COOP=1 VT_NORM_QUANT_FUSED=1 VT_RMSNORM_ROW_COOP=1 VT_GDN_NORMGATED_COOP=1 VT_GDN_POSTCONV_COOP=1 VT_GDN_SCAN_SPLIT=1 VT_ARGMAX_SPLIT=1 VT_GDN_ROWPERM_KEEP_QUANT=1 VT_RMSNORM_LDS_QUANT=1 VT_GDN_COLPERM_KEEP_QUANT=1 VT_QUANT_Q8K_WARP=1"
ENVS=""
ARGS=""
seen_dashdash=0
for a in "$@"; do
  if [ "$seen_dashdash" -eq 0 ]; then
    if [ "$a" = "--" ]; then seen_dashdash=1; continue; fi
    if [ "$a" = "@levers" ]; then
      # The adopted-lever block, verbatim from the T33/T34 reference runs
      # (agent-artifacts tg200-t33-eval / tg200-t34-capture inner scripts).
      ENVS="$ENVS $LEVERS"
      continue
    fi
    case "$a" in
      *=*) ENVS="$ENVS $a" ;;
      *) echo "tg200-neartie.sh: non KEY=VALUE token before -- : $a" >&2; exit 2 ;;
    esac
  else
    ARGS="$ARGS \"$a\""
  fi
done
if [ "$seen_dashdash" -eq 0 ]; then
  echo "usage: tools/tg200-neartie.sh <capture|adjudicate> [KEY=VALUE|@levers ...] -- [binary args...]" >&2
  exit 2
fi

echo "== gpu-ctl status =="
/home/ghazni/gpu-coord/gpu-ctl status || true

# The engine load + decode are GPU work: acquire with a generous timeout and
# let gpu-ctl serialize against the co-tenant rather than polling.
exec /home/ghazni/gpu-coord/gpu-ctl acquire 1800 "TG200 near-tie $MODE" -- \
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    -u 1000:1000 -e HOME=/job -w /job \
    -e TG200_NEARTIE_DEBUG="${TG200_NEARTIE_DEBUG:-0}" \
    -v /home/ghazni/github/vllm.cpp/tg200:/repo/tg200 \
    -v /home/ghazni/models:/models \
    rocm-dev:10.0.0 \
    env LD_LIBRARY_PATH=/opt/rocm/lib \
    /bin/sh -c "
      set -eu
      for kv in $ENVS; do export \"\$kv\"; done
      echo '== levers in effect ==' ; env | grep -E '^VT_' | sort || true
      exec /repo/tg200/build-hip-docker/examples/tg200-neartie $MODE $ARGS
    "
