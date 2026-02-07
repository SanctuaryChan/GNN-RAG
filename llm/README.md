## Get Started
We have simple requirements in `requirements.txt`. You can always check if you can run the code immediately.

Please also download `entities_names.json` file from https://drive.google.com/drive/folders/1ifgVHQDnvFEunP9hmVYT07Y3rvcpIfQp?usp=sharing, as GNNs use the dense graphs. 

## Evaluation
We provide the results of GNN retrieval in `results/gnn`. To evaluate GNN-RAG performance, run `scripts/rag-reasoning.sh`. 

You can also compute perfromance on multi-hop question by `scripts/evaluate_multi_hop.sh`. 

To test different LLMs for KGQA (ChatGPT, LLaMA2), see `scripts/plug-and-play.sh`. 

## Path Verbalizer & Reranker (Optional)
You can improve path quality before feeding to LLMs by using a template verbalizer and a path reranker.

1) Build reranker dataset:
```
python -m align_kg.build_path_rank_dataset --data_path rmanluo -d webqsp --split train --use_verbalizer --entities_names_path /path/to/entities_names.json
```

2) Train a cross-encoder reranker:
```
python -m align_kg.train_path_reranker --data_path datasets/PathRankData/webqsp/webqsp_train.jsonl --output_dir saved_models/path_reranker --mode cross --use_answer --use_question
```

3) Run prediction with verbalizer + reranker:
```
python -m qa_prediction.predict_answer --use_verbalizer --reranker_path saved_models/path_reranker --reranker_topk 5 --entities_names_path /path/to/entities_names.json
```

## Resutls

We append all the results for Table 2: See `results/KGQA-GNN-RAG-RA`. You can look at the actual LLM generations, as well as the KG information retrieved ("input" key) in `predictions.jsonl`.



## 1) 构建 reranker 训练集

```shell
python -m align_kg.build_path_rank_dataset \
--data_path rmanluo \
-d webqsp \
--split train \
--use_verbalizer \
--hop 2 \
--n_neg 2 \
--output_path datasets/PathRankData \
--save_name webqsp_train.jsonl \
--seed 42
```

  参数解释

  - --data_path rmanluo
    数据集根目录（和你现在 pipeline 的 WebQSP 数据位置一致）
  - -d webqsp
    数据集名称（会拼成 rmanluo/webqsp）
  - --split train
    用训练集生成 reranker 数据
  - --use_verbalizer
    用模板 verbalizer 生成路径文本（推荐保持和推理一致）
  - --hop 2
    负样本路径的最大长度（随机游走 hop）
  - --n_neg 2
    每个样本的负路径数量（可调大但会变慢）
  - --output_path datasets/PathRankData
    输出文件夹
  - --save_name webqsp_train.jsonl
    输出文件名
  - --seed 42
    固定随机性，便于复现

  输出文件路径：
  datasets/PathRankData/webqsp/webqsp_train.jsonl

  ———

  ## 2) 训练 cross‑encoder reranker

```shell
python -m align_kg.train_path_reranker \
--data_path datasets/PathRankData/webqsp/webqsp_train.jsonl \
--output_dir saved_models/path_reranker \
--mode cross \
--model_name_or_path bert-base-uncased \
--epochs 3 \
--batch_size 16 \
--lr 2e-5 \
--max_length 256 \
--warmup_ratio 0.1 \
--use_answer \
--use_question
```
  参数解释

  - --data_path
    上一步生成的 jsonl
  - --output_dir
    保存 reranker 模型的目录
  - --mode cross
    使用 cross‑encoder 训练（最稳、效果好）
  - --model_name_or_path bert-base-uncased
    基座模型（可换成 distilbert-base-uncased 或 microsoft/MiniLM-L6-H384-uncased）
  - --epochs 3
    训练轮数
  - --batch_size 16
    batch 大小（视显存调整）
  - --lr 2e-5
    学习率
  - --max_length 256
    输入截断长度
  - --warmup_ratio 0.1
    warmup 比例
  - --use_answer
    训练时文本拼接中包含候选答案 [ANS]
  - --use_question
    训练时文本拼接中包含问题 [Q]

  输出模型会在：
  saved_models/path_reranker/

  ———

  ## 3) 推理时启用 verbalizer + reranker

```shell
python -m qa_prediction.predict_answer \
--use_verbalizer \
--verbalizer_mode plain \
--verbalizer_operator auto \
--reranker_path saved_models/path_reranker \
--reranker_mode cross \
--reranker_topk 5 \
--reranker_max_length 256 \
--reranker_use_answer \
--reranker_use_question
```
  参数解释

  - --use_verbalizer
    推理时也使用模板 verbalizer（确保训练/推理一致）
  - --verbalizer_mode plain
    plain=普通模板；answer=带“此路径支持答案…”的模板
  - --verbalizer_operator auto
    自动从问题文本推断 when/where/who 等算子
  - --reranker_path
    reranker 模型路径
  - --reranker_mode cross
    使用 cross‑encoder reranker
  - --reranker_topk 5
    仅保留 top‑5 路径进入 prompt
  - --reranker_max_length 256
    reranker 输入截断
  - --reranker_use_answer
    reranker 输入拼接 [ANS]
  - --reranker_use_question
    reranker 输入拼接 [Q]

  ———

  ## 常见调参建议

  - 路径太多/太长：调小 --reranker_topk 或 --max_length
  - 排序质量不足：加大 --n_neg 或换更强 backbone
  - 训练慢：用 distilbert 或 MiniLM，调小 --epochs
  - 想用 alignment 模式：训练时 --mode align，推理时 --reranker_mode align

  ———
