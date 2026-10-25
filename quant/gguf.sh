#!/usr/bin/env bash
# Convert a Hugging Face checkpoint to GGUF F16 with llama.cpp, then quantize it.
#
# usage: quant/gguf.sh <hf_model_dir> <out_dir> [quant_type]
# needs: LLAMA_CPP_DIR pointing at a llama.cpp checkout built with llama-quantize,
#        and the Python requirements of convert_hf_to_gguf.py installed.
# writes: <out_dir>/<name>-F16.gguf, <out_dir>/<name>-<quant>.gguf, <out_dir>/timings.json
set -euo pipefail

HF_DIR=${1:?hf model dir}
OUT_DIR=${2:?output dir}
QUANT=${3:-Q4_K_M}
NAME=${GGUF_NAME:-Qwen3-8B}
LLAMA_CPP_DIR=${LLAMA_CPP_DIR:?set LLAMA_CPP_DIR}

mkdir -p "$OUT_DIR"
F16="$OUT_DIR/$NAME-F16.gguf"
QOUT="$OUT_DIR/$NAME-$QUANT.gguf"

t0=$(date +%s)
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$HF_DIR" --outtype f16 --outfile "$F16"
t1=$(date +%s)
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$F16" "$QOUT" "$QUANT"
t2=$(date +%s)

cat > "$OUT_DIR/timings.json" <<EOF
{"convert_s": $((t1 - t0)), "quantize_s": $((t2 - t1)), "f16_bytes": $(stat -c %s "$F16"), "quant_bytes": $(stat -c %s "$QOUT")}
EOF
echo "done: $QOUT"
