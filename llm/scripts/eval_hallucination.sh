#!/usr/bin/env bash
set -euo pipefail

# 幻觉评估脚本：从 predictions.jsonl + GNN test.info 计算 EC/HR/SH 等指标
# 使用方式：
#   bash test_scripts/eval_hallucination.sh
#   bash test_scripts/eval_hallucination.sh /path/to/predictions.jsonl /path/to/test.info
#
# 可选环境变量：
#   TEST_JSON=...          # 指定 test.json（默认与 test.info 同目录）
#   ENTITIES_NAMES=...     # 指定 entities_names.json（默认 ../entities_names.json）
#   MAX_LINE_LEN=80        # 过滤解释性行的最大长度
#   EXPLAIN_KEYWORDS=...   # 解释性关键词（英文逗号分隔）
#   CONTAINMENT=1          # 开启子串包含匹配（更宽松，可能带来误报）
#   MAX_CAND=0             # 截断候选实体数量（0 表示不截断）
#   OUTPUT_DIR=...         # 指定输出目录（默认与 predictions.jsonl 同目录）
#   KEEP_ONLY_CAND=1       # 只保留命中候选实体的片段
#   DROP_EXPLAIN_UNLESS_CAND=1 # 解释行默认丢弃，除非包含候选实体

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_DIR="$(dirname "${SCRIPT_DIR}")"

DATASET_NAME="webqsp"
MODAL_NAME="llama2"

# 默认路径（可通过命令行参数覆盖）
DEFAULT_PREDICTIONS="${LLM_DIR}/results/KGQA-GNN-RAG/rearev-sbert/RoG-${DATASET_NAME}/${MODAL_NAME}/test/no_rule/False/predictions.jsonl"
DEFAULT_GNN_INFO="${LLM_DIR}/results/gnn/RoG-${DATASET_NAME}/rearev-sbert/test.info"

PREDICTIONS=${1:-"${DEFAULT_PREDICTIONS}"}
GNN_INFO=${2:-"${DEFAULT_GNN_INFO}"}

TEST_JSON=${TEST_JSON:-""}
ENTITIES_NAMES=${ENTITIES_NAMES:-"${LLM_DIR}/entities_names.json"}
MAX_LINE_LEN=${MAX_LINE_LEN:-80}
EXPLAIN_KEYWORDS=${EXPLAIN_KEYWORDS:-"based on,possible answers,therefore,because,explanation,so,answer is,the answer"}
CONTAINMENT=${CONTAINMENT:-1}
KEEP_ONLY_CAND=${KEEP_ONLY_CAND:-1}
DROP_EXPLAIN_UNLESS_CAND=${DROP_EXPLAIN_UNLESS_CAND:-1}

MAX_CAND=${MAX_CAND:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"$(dirname "${PREDICTIONS}")"}

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DETAIL_FILE="${OUTPUT_DIR}/hallucination_eval_detail.jsonl"
OUTPUT_RESULT_FILE="${OUTPUT_DIR}/hallucination_eval_result.jsonl"

CONTAINMENT_FLAG=""
if [[ "${CONTAINMENT:-0}" == "1" ]]; then
  CONTAINMENT_FLAG="--containment"
fi

KEEP_ONLY_CAND_FLAG=""
if [[ "${KEEP_ONLY_CAND:-0}" == "1" ]]; then
  KEEP_ONLY_CAND_FLAG="--keep-only-cand"
fi

DROP_EXPLAIN_UNLESS_CAND_FLAG=""
if [[ "${DROP_EXPLAIN_UNLESS_CAND:-0}" == "1" ]]; then
  DROP_EXPLAIN_UNLESS_CAND_FLAG="--drop-explain-unless-cand"
fi

TEST_JSON_FLAG=""
if [[ -n "${TEST_JSON}" ]]; then
  TEST_JSON_FLAG="--test-json ${TEST_JSON}"
fi

python scripts/eval_hallucination.py \
  --predictions "${PREDICTIONS}" \
  --gnn-info "${GNN_INFO}" \
  ${TEST_JSON_FLAG} \
  --entities-names "${ENTITIES_NAMES}" \
  --output "${OUTPUT_DETAIL_FILE}" \
  --summary-output "${OUTPUT_RESULT_FILE}" \
  --max-line-len "${MAX_LINE_LEN}" \
  --explain-keywords "${EXPLAIN_KEYWORDS}" \
  --max-cand "${MAX_CAND}" \
  --log-samples 10 \
  --log-path /tmp/hallu_${DATASET_NAME}_debug.jsonl \
  ${CONTAINMENT_FLAG} \
  ${KEEP_ONLY_CAND_FLAG} \
  ${DROP_EXPLAIN_UNLESS_CAND_FLAG}

