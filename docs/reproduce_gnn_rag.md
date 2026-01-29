# GNN-RAG 实验复现指南（以本仓库实现为准）

本文档聚焦复现 **GNN-RAG** 及其 RA 变体（GNN-RAG+RA）的实验流程。论文中部分对比方法（如 ToG、SubgraphRAG 等）多为引用原论文结果，本文档不要求完整复现这些 baseline。

## 1. 目录结构速览
- `gnn/`：GNN 检索器训练与评估（ReaRev / NSM / GraftNet 等）。
- `llm/`：RAG 生成答案、评估、多跳分析。
- `llm/results/`：仓库内包含部分预计算结果（GNN 检索输出、RoG 路径、GNN-RAG 输出）。

## 2. 环境与依赖
建议将 GNN 与 LLM 分为两个 Python 环境（依赖版本差异较大）。

### 2.1 GNN 环境
```bash
python -m venv .venv-gnn
source .venv-gnn/bin/activate
pip install -r gnn/requirements.txt
```
- `gnn/requirements.txt` 里固定了 `torch==1.7.1+cu110`，如果你的 CUDA 版本不同，请替换为匹配版本。

### 2.2 LLM 环境
```bash
python -m venv .venv-llm
source .venv-llm/bin/activate
pip install -r llm/requirements.txt
```
- 需要 `transformers`, `datasets`, `peft`, `accelerate`, `networkx` 等。
- 如使用 ChatGPT，需要设置 `OPENAI_API_KEY`。

## 3. 数据与模型准备
### 3.1 GNN 数据（WebQSP / CWQ / MetaQA）
GNN 训练数据与预训练 LM（LMsr）见仓库 README 的 Google Drive 链接：
- https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing
- 包含：`data.zip`、`entities_names.json`、`pretrained_lms.zip`

解压后建议放置为：
```
GNN-RAG/
  gnn/
    data/
      webqsp/
        train.json
        valid.json
        test.json
        vocab.txt
        relations.txt
        entities.txt
        chars.txt
        word_emb.npy
        rel_word_idx.npy
      CWQ/
        train.json
        valid.json
        test.json
        ...
```
以上文件名是 `gnn/parsing.py` 中的默认配置。

### 3.2 LLM 侧数据
- `entities_names.json`：用于实体 ID 与名称映射（仓库已自带在 `llm/`，若需更新也在同一 Google Drive 链接中）。
- RoG 数据集：脚本默认从 HuggingFace 加载 `rmanluo/RoG-webqsp` 与 `rmanluo/RoG-cwq`。
  - `llm/src/qa_prediction/predict_answer.py` 与 `gen_rule_path.py` 都通过 `datasets.load_dataset("rmanluo/RoG-webqsp")` 方式加载。

### 3.3 LLM 模型
- `rmanluo/RoG`：RoG 的 LLM（7B Llama2 微调），脚本默认使用它。
- `meta-llama/Llama-2-7b-chat-hf`：作为基础模型或替换。

注意：
- `llm/src/llms/language_models/llama.py` 与 `llm/src/qa_prediction/gen_rule_path.py` 中 **硬编码了 HuggingFace token**。
- 建议你将 `token=...` 替换为自己的 token，或删除并使用 `huggingface-cli login`。

### 3.4 为什么 GNN 与 LLM 数据源不同？如何对齐？
- **同一任务，不同数据格式**：
  - GNN 侧需要“稠密子图 + 训练字典/嵌入”（适配图神经网络训练），所以数据由作者预处理后放在 Google Drive。
  - LLM 侧需要“问题 + 图三元组 + q_entity/a_entity”等（用于构造 RAG prompt 与评估），该格式来自 RoG，因此用 HuggingFace `rmanluo/RoG-*` 直接加载。
- **对齐原则**：两侧必须对齐到同一批问题 ID，GNN 的 `test.info` 会按行与 `test.json` 对齐，`predict_answer.py` 会用 `id` 进行匹配。
- **实践建议**：
  1) 从 GNN 数据目录复制对应的 `test.json` 到 GNN 输出目录（见第 5 节）。
  2) 使用仓库自带的 `llm/results/gnn/*/test.info` 时，也要确保配套的 `test.json` 存在。
  3) 如需脱离 HuggingFace 在线加载，可用 `datasets` 先下载到本地，再把 `--data_path` 指向本地目录（`predict_answer.py` 与 `gen_rule_path.py` 的 `--data_path` 参数）。

## 4. GNN 检索器训练与评估（可选）
如果你不想训练 GNN，可跳过本节，直接使用仓库中 `llm/results/gnn/*/test.info`。

### 4.1 WebQSP 训练示例
```bash
cd gnn
python main.py ReaRev \
  --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 \
  --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 3 \
  --relation_word_emb True --name webqsp --experiment_name prn_webqsp_rearev_sbert
```

### 4.2 WebQSP 评估并导出检索结果
```bash
cd gnn
python main.py ReaRev \
  --entity_dim 50 --num_epoch 200 --batch_size 8 --eval_every 2 \
  --data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 3 \
  --relation_word_emb True --load_experiment prn_webqsp_rearev_sbert.ckpt \
  --is_eval --name webqsp --experiment_name prn_webqsp_rearev_sbert
```
输出文件默认在 `gnn/checkpoint/pretrain/prn_webqsp_rearev_sbert_test.info`。

### 4.3 CWQ 评估示例
脚本在 `gnn/scripts/rearev_cwq.sh` 中：
```bash
cd gnn
python main.py ReaRev \
  --entity_dim 50 --num_epoch 100 --batch_size 8 --eval_every 2 \
  --data_folder data/CWQ/ --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 \
  --relation_word_emb True --load_experiment ReaRev_CWQ.ckpt \
  --is_eval --name cwq --experiment_name prn_cwq_rearev_sbert
```

## 5. 将 GNN 输出接入 LLM
`llm/src/qa_prediction/predict_answer.py` 需要 **GNN 输出 + 对应 test.json** 放在同一目录。

以 WebQSP 为例：
```bash
mkdir -p llm/results/gnn/RoG-webqsp/rearev-sbert
cp gnn/checkpoint/pretrain/prn_webqsp_rearev_sbert_test.info \
  llm/results/gnn/RoG-webqsp/rearev-sbert/test.info
cp gnn/data/webqsp/test.json \
  llm/results/gnn/RoG-webqsp/rearev-sbert/test.json
```
CWQ 同理：
```
llm/results/gnn/RoG-cwq/rearev-sbert/test.info
llm/results/gnn/RoG-cwq/rearev-sbert/test.json
```

## 6. RoG 关系路径（RA 用）
RA 需要 RoG 生成的 relation paths。仓库已提供部分结果：
- `llm/results/gen_rule_path/RoG-webqsp/RoG/test/predictions_3_False.jsonl`
- `llm/results/gen_rule_path/RoG-cwq/RoG/test/predictions_3_False.jsonl`

如需重新生成：
```bash
cd llm
bash scripts/planning.sh
```
- 默认只跑 WebQSP，如需 CWQ 请修改脚本中的 `DATASET_LIST`。
- 需要可用的 HuggingFace 模型与 token。

## 7. 运行 GNN-RAG（核心实验）
### 7.1 GNN-RAG（不使用 RA）
```bash
cd llm
bash scripts/rag-reasoning.sh
```
默认设置：
- 数据集：`RoG-webqsp`
- LLM：`rmanluo/RoG`
- 结果输出：`llm/results/KGQA-GNN-RAG/rearev-sbert/.../predictions.jsonl`

如需 CWQ，请修改 `scripts/rag-reasoning.sh` 中 `DATASET_LIST`。

### 7.2 GNN-RAG+RA
在 `scripts/rag-reasoning.sh` 中取消注释 RA 段落，或直接运行：
```bash
cd llm
python src/qa_prediction/predict_answer.py \
  --model_name RoG \
  -d RoG-webqsp \
  --prompt_path prompts/llama2_predict.txt \
  --add_rule \
  --rule_path results/gen_rule_path/RoG-webqsp/RoG/test/predictions_3_False.jsonl \
  --rule_path_g1 results/gnn/RoG-webqsp/rearev-sbert/test.info \
  --rule_path_g2 None \
  --model_path rmanluo/RoG \
  --predict_path results/KGQA-GNN-RAG-RA/rearev-sbert
```

## 8. 评估与输出
`predict_answer.py` 在生成 `predictions.jsonl` 后会自动调用 `evaluate_results.py`，生成：
- `eval_result.txt`
- `detailed_eval_result.jsonl`

多跳问题统计：
```bash
cd llm
bash scripts/evaluate_multi_hop.sh
```
（脚本内的路径默认指向 WebQSP 结果目录，可按需修改。）

## 9. 已有结果与对照
仓库已提供复现结果，可直接对比：
- `llm/results/KGQA-GNN-RAG/*`（GNN-RAG）
- `llm/results/KGQA-GNN-RAG-RA/*`（GNN-RAG+RA）

这些结果对应论文表 2 的主要实验。

## 10. 常见问题
1. **找不到 `entities_names.json`**：必须从 `llm/` 目录运行脚本，或把文件复制到当前工作目录。
2. **`test.json` 缺失**：`predict_answer.py` 会从 `test.info` 同目录读取 `test.json`，请务必补齐。
3. **HuggingFace token 报错**：请替换源码中硬编码 token 或使用 `huggingface-cli login`。
4. **CUDA / torch 版本不匹配**：建议按本机 CUDA 版本重装 torch。

如果你希望我进一步补充“单机最小复现配置”、“在无 GPU 环境下的替代流程”，或“针对 WebQSP/CWQ 的完整命令串联”，告诉我你当前的硬件与目标即可。
