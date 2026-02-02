# Graph-Constrained Decoding (GCD) 设计文档

本文档基于当前 GNN-RAG 代码结构，提出一个“即插即用式”的 Graph-Constrained Decoding（GCD）模块设计，用于在生成阶段利用检索子图/路径对 LLM 输出进行结构化约束。

## 1. 目标与非目标

**目标**
- 在不改检索器（GNN/RoG/SubgraphRAG）的前提下，通过解码约束提高答案一致性与准确率。
- 作为可选模块接入：默认不改变现有行为，启用时才生效。
- 支持 hard constraint（硬约束）与 soft constraint（软惩罚）两种方式。

**非目标**
- 不训练新的大型模型；可选引入极小的“校准器/阈值学习器”。
- 不改变现有数据格式；尽量复用现有 `test.info` 与 `predicted_paths`。

## 2. 现有流程与可插入点

当前 LLM 推理路径如下：
1) `predict_answer.py` 从 `args.rule_path_g1` 读取 GNN 检索输出，并把候选实体写入 `data["cand"]`。
2) `PromptBuilder.process_input()` 基于候选实体生成 reasoning paths 文本并拼接到 prompt。
3) `Llama.generate_sentence()` 直接用 `transformers.pipeline(text-generation)` 解码，输出自然语言答案。

GCD 的最佳插入点：
- **解码阶段**：在 `BaseLanguageModel.generate_sentence()` 内部注入 token-level 约束。
- **输入准备阶段**：在 `predict_answer.py` 里提取 `data["cand"]` 与 `predicted_paths`，构造约束条件对象，传给 LLM。

## 3. 约束来源与规则

### 3.1 约束来源
- **GNN 候选实体**：`data["cand"]`（来自 `test.info`），覆盖 top-K 实体。适合“实体答案约束”。
- **RoG 关系路径**（RA）：`predicted_paths`，可衍生“关系集合”或“路径 token”。适合“关系/路径答案约束”。
- **Schema 级别**：从 KG schema 或关系词表生成允许的 relation 名称集合。

### 3.2 约束类型
- **实体约束**：仅允许输出候选实体名（或其别名）。
- **关系约束**：仅允许输出候选关系名。
- **路径约束**：限制输出为合法路径 token 序列（更严格）。

### 3.3 约束强度
- **Hard**：完全屏蔽不在 Allowed Set 中的 token（logits = -inf）。
- **Soft**：对 Allowed Set 外 token 施加惩罚（logits -= lambda）。

## 4. 模块化设计（即插即用）

### 4.1 新增模块结构
建议新增目录：
```
llm/src/constraints/
  __init__.py
  spec.py              # 约束描述结构（ConstraintSpec）
  entity_vocab.py      # 实体名/别名处理与规范化
  trie.py              # Trie / PrefixTree
  logits.py            # LogitsProcessor / LogitsWarper
  builder.py           # 从 cand / paths 构造约束
```

### 4.2 关键数据结构
**ConstraintSpec**
- `mode`: none | entity | relation | path
- `strength`: hard | soft
- `allowed_sequences`: List[List[int]]  # 允许的 token 序列（Trie 输入）
- `allowed_tokens`: Optional[Set[int]]
- `lambda`: Optional[float]             # soft penalty 强度
- `debug_info`: Optional[dict]

### 4.3 约束构建器
**ConstraintBuilder**
输入：
- `cand_entities`（来自 `data["cand"]`）
- `predicted_paths`（可选）
- tokenizer
输出：
- `ConstraintSpec`（包含 Trie/allowed tokens）

关键步骤：
1) 实体标准化：使用 `entities_names.json` 映射为文本名，统一大小写/标点格式。
2) 生成允许序列：tokenizer 编码实体名，构建 Trie。
3) 软约束时仅产出 allowed token set 或 token frequency prior。

### 4.4 解码注入
- **Hard constraint**：用 `prefix_allowed_tokens_fn` 或自定义 `LogitsProcessor` 过滤 token。
- **Soft constraint**：自定义 `LogitsProcessor` 给不在 allowed set 的 token 加惩罚。

注意：`transformers.pipeline()` 无法直接注入 `LogitsProcessor`，需要改为 `AutoModelForCausalLM.generate()`。因此建议对 `Llama` 类做一条“受约束解码分支”，保持无约束路径不变。

## 5. 代码改动建议（保持最小侵入）

### 5.1 BaseLanguageModel 扩展
- 新增可选接口：
  - `generate_sentence(lm_input, constraints: Optional[ConstraintSpec] = None)`
- 默认实现保持兼容。

### 5.2 Llama 实现扩展
- 保留现有 pipeline 逻辑作为默认路径。
- 当传入 `constraints` 时，改用 `AutoModelForCausalLM.generate()`，注入自定义 `LogitsProcessorList`。

### 5.3 predict_answer.py 增强
- 在 `prediction()` 内部，从 `data["cand"]` 与 `predicted_paths` 生成 `ConstraintSpec`。
- 通过 LLM 接口传入约束。

### 5.4 PromptBuilder 轻量配合
- 启用 `--each_line`，并在 prompt 中明确“只输出答案列表，禁止解释”。
- 可新增 `prompts/llama2_constrained.txt` 作为严格输出版本。

## 6. 指标与验证

### 6.1 主要指标
- Answer-F1 / Hits@1 / Accuracy（已有评估脚本）

### 6.2 幻觉相关指标（与 `eval_hallucination.py` 一致）
**可用字段**
- `predictions.jsonl`: `id`, `prediction`（原始模型输出）
- `results/gnn/.../test.info`: `cand`（GNN 检索候选实体）
- `entities_names.json`: `mid -> name` 映射（用于把 `cand` 转为实体名）

**集合与符号定义**
- 第 \(i\) 个样本候选实体集合：\(C_i\)。由 `cand` 的 `mid` 映射到实体名后再 normalize。
- 第 \(i\) 个样本预测实体集合：\(P_i\)。从 `prediction` 抽取片段并 normalize 后得到集合。
- 归一化函数 \( \mathrm{norm}(\cdot) \)：小写、去标点、去冠词、去多余空白（与 `evaluate_results.py` 一致）。

**预测抽取规则（脚本当前实现）**
1. 按行切分 `prediction`，去掉行首编号/符号（如 `1.`、`-`）。  
2. 每行按分隔符切片（`,` `;`，以及短句中的 `and`）。  
3. 解释行判定：若行中包含解释关键词，则记 \(Explain_i=1\)。  
4. 行/片段过滤：  
   - 启用 `--drop-explain-unless-cand` 时，解释行默认丢弃，除非该行包含候选实体。  
   - 启用 `--keep-only-cand` 时，只保留命中候选实体的片段。  
   - 片段长度超过阈值 `max_line_len` 时，仅在“命中候选”时保留。  
   - “命中候选”支持 **exact** 与 **containment**（`--containment` 开启子串包含）。  

**指标定义（逐样本）**  
令 \(P_i\) 为预测实体集合、\(C_i\) 为候选实体集合：
- **Evidence Consistency (EC)**  
  \[
  EC_i = \frac{|P_i \cap C_i|}{\max(1, |P_i|)}
  \]
- **Hallucination Rate (HR)**  
  \[
  HR_i = 1 - EC_i
  \]
- **Strict Hallucination (SH)**  
  \[
  SH_i = \mathbb{1}\left(|P_i \setminus C_i| > 0\right)
  \]
- **Empty**  
  \[
  Empty_i = \mathbb{1}\left(|P_i| = 0\right)
  \]
- **ExplainHit**  
  \[
  Explain_i = \mathbb{1}\left(\exists\ \text{行含解释关键词}\right)
  \]

**汇总统计（脚本输出）**  
对可对齐候选的样本集合 \(\mathcal{D}\) 取均值：
\[
EC = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} EC_i,\quad
HR = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} HR_i,\quad
SH = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} SH_i
\]
\[
EmptyRate = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} Empty_i,\quad
ExplainRate = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} Explain_i
\]

**实现注意**
- 这些指标不依赖 `ground_truth`，只衡量“输出是否落在候选证据内”。  
- 若 \(P_i\) 为空，脚本返回 \(EC_i=0, HR_i=0, SH_i=0\)，并在 `EmptyRate` 中统计。  
- 启用 `--keep-only-cand` 与 `--drop-explain-unless-cand` 后，指标更接近“实体级幻觉”而非“解释文本噪声”。  

### 6.3 消融实验
- hard vs soft
- 不同 top-K 候选数
- 仅实体约束 vs 关系/路径约束

## 7. 失败模式与兜底策略

- **候选集不含真答案**：硬约束会必错；建议使用 soft 或 fallback 机制。
- **实体名多 token**：Trie 必须支持多 token 组合；避免只做单 token allowlist。
- **输出冗长**：prompt 强化 + `--each_line` + 低 `max_new_tokens`。

## 8. 配置建议（即插即用）

新增 CLI 参数（建议）：
```
--constraint_mode {none,entity,relation,path}
--constraint_strength {hard,soft}
--constraint_k 50
--constraint_source {gnn,rog,union}
--constraint_lambda 2.0
--constraint_debug
```

运行示例：
```
python src/qa_prediction/predict_answer.py \
  --model_name llama \
  --model_path /data/GNN-RAG/models/Llama-2-7b-chat-hf \
  -d RoG-webqsp \
  --constraint_mode entity \
  --constraint_strength hard \
  --constraint_k 50
```

## 9. 里程碑（最小可用版本）

1) 只做实体约束（hard/soft）
2) 集成到 `Llama.generate_sentence()`
3) 新增评估指标（evidence consistency）
4) 扩展到关系/路径约束

---

如需我进一步把该设计落地成代码改动清单（PR 级别）或提供最小可运行实现模板，告诉我你想先做 hard 还是 soft 版本。
