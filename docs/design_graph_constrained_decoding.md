# Graph-Constrained Decoding (GCD) 设计文档

本文档基于当前 GNN-RAG 代码结构，提出一个“即插即用式”的 Graph-Constrained Decoding（GCD）模块设计，用于在生成阶段利用检索子图/路径对 LLM 输出进行结构化约束，并确保接口与具体检索/路由技术解耦，便于迁移到其他相关代码库。

## 1. 目标与非目标

**目标**
- 在不改检索器（任意 KG 检索器/子图检索器）的前提下，通过解码约束提高答案一致性与准确率。
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

### 3.1 约束来源（与本项目解耦）
为保证模块“即插即用”，约束来源被设计为**可插拔 Provider**，不依赖任何特定检索器/路由器实现。GCD 只要求上游输出满足统一的**候选/证据协议**即可。

**统一输入协议（ConstraintInputs）**
- `candidates`: List[str]  # 候选实体表（surface forms）
- `evidence_paths`: Optional[List[List[Tuple[str, str, str]]]]  # 证据路径/三元组序列
- `evidence_relations`: Optional[List[str]]  # 可选，显式关系集合
- `extra_candidates`: Optional[List[str]]  # 可选，外部扩展候选

**典型来源（按强到弱，可做消融）**
- **S1 Answer Candidates（强约束）**：上游检索器给出的候选实体集合。  
  在本项目中可直接来自 `data["cand"]`（由 `test.info` 生成），但在其他项目中也可以来自任意 retriever 的候选列表。
- **S2 Evidence Nodes/Triples（中约束）**：证据路径/三元组中出现的实体与关系。  
  在本项目中可由 `predicted_paths` 或 `get_truth_paths()` 产出的路径推得；在其他系统中可由检索器输出的 evidence triples 提供。
- **S3 Extra/External Candidates（弱约束，提升召回）**：来自其他检索模块/长上下文/规则系统的补充候选。  
  这是一个“可选扩展位”，GCD 不要求其具体来源，也不与任何特定路由技术绑定。

**结论**：GCD 不依赖任何项目专有检索/路由技术；只要将任意上游输出映射到 `ConstraintInputs`，即可复用。

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
  builder.py           # 从 ConstraintInputs 构造约束
  providers/
    __init__.py
    base.py            # 抽象 Provider 接口（解耦用）
    project_adapter.py # （可选）本项目适配器
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
- `ConstraintInputs`（统一候选/证据协议）
- tokenizer
输出：
- `ConstraintSpec`（包含 Trie/allowed tokens）

说明：在本项目里，`ConstraintInputs` 可由 `data["cand"]` 和 `predicted_paths` 组装；在其他项目里由对应 Provider 适配即可。

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
- 在 `prediction()` 内部，通过本项目的 Provider/Adapter 把 `data["cand"]` 与 `predicted_paths` 映射为 `ConstraintInputs`。
- 由 `ConstraintBuilder` 生成 `ConstraintSpec` 并传入 LLM。

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
- 第 $i$ 个样本候选实体集合：$C_i$。由 `cand` 的 `mid` 映射到实体名后再 normalize。
- 第 $i$ 个样本预测实体集合：$P_i$。从 `prediction` 抽取片段并 normalize 后得到集合。
- 归一化函数 $\\mathrm{norm}(\\cdot)$：小写、去标点、去冠词、去多余空白（与 `evaluate_results.py` 一致）。

**预测抽取规则（脚本当前实现）**
1. 按行切分 `prediction`，去掉行首编号/符号（如 `1.`、`-`）。  
2. 每行按分隔符切片（`,` `;`，以及短句中的 `and`）。  
3. 解释行判定：若行中包含解释关键词，则记 $Explain_i=1$。  
4. 行/片段过滤：  
   - 启用 `--drop-explain-unless-cand` 时，解释行默认丢弃，除非该行包含候选实体。  
   - 启用 `--keep-only-cand` 时，只保留命中候选实体的片段。  
   - 片段长度超过阈值 `max_line_len` 时，仅在“命中候选”时保留。  
   - “命中候选”支持 **exact** 与 **containment**（`--containment` 开启子串包含）。  

**指标定义（逐样本）**  
令 $P_i$ 为预测实体集合、$C_i$ 为候选实体集合：
- **Evidence Consistency (EC)**  
  $$
  EC_i = \frac{|P_i \cap C_i|}{\max(1, |P_i|)}
  $$
  中文释义：**证据一致性**。衡量“模型输出的候选答案里，有多少比例能在检索候选集合中找到支撑”。  
  解读：
  - $EC_i$ 越高，说明模型越“听话”，输出越贴近检索证据（更少凭空编造）。  
  - 若你启用了 `--keep-only-cand`，则 $EC_i$ 往往会显著提高（因为抽取阶段只保留命中候选的片段）。  
  - $EC_i$ 不等价于正确率：候选集合里可能包含很多“证据内但不正确”的实体。
- **Hallucination Rate (HR)**  
  $$
  HR_i = 1 - EC_i
  $$
  中文释义：**幻觉率（比例）**。表示“被抽取出来的预测片段中，有多少比例不在候选证据集合里”。  
  解读：
  - $HR_i$ 越高，说明模型越容易输出候选证据外的内容（更可能出现幻觉）。  
  - 该定义是“候选证据外”意义上的幻觉，不直接验证是否与 `ground_truth` 一致。
- **Strict Hallucination (SH)**  
  $$
  SH_i = \mathbb{1}\left(|P_i \setminus C_i| > 0\right)
  $$
  中文释义：**严格幻觉（是否发生）**。只要预测集合中出现任意一个候选外片段，就记为 1。  
  解读：
  - $SH_i$ 更像“是否触发过幻觉”的告警指标；它对解释性输出/冗余输出非常敏感。  
  - 适合用来比较“约束解码是否能把候选外输出压到几乎为 0”，但不适合衡量“幻觉的量有多大”（因为只要出现一次就为 1）。
- **Empty**  
  $$
  Empty_i = \mathbb{1}\left(|P_i| = 0\right)
  $$
  中文释义：**空答案指示**。表示该样本的输出中没有抽取到任何可用于匹配的实体片段。  
  解读：
  - 可能原因：模型没有按要求输出列表；输出全是长解释导致被过滤；或抽取规则过严。
- **ExplainHit**  
  $$
  Explain_i = \mathbb{1}\left(\exists\ \text{行含解释关键词}\right)
  $$
  中文释义：**解释命中**。只要原始输出中出现解释性关键词，就记为 1（不要求该行最终被用于计算 $P_i$）。  
  解读：
  - 这是一个“行为统计”指标，用于判断模型是否经常输出解释/链式推理等冗余文本。  
  - 该指标不等价于幻觉，只反映“是否倾向解释”。

**汇总统计（脚本输出）**  
对可对齐候选的样本集合 $\mathcal{D}$ 取均值：
$$
EC = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} EC_i,\quad
HR = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} HR_i,\quad
SH = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} SH_i
$$
$$
EmptyRate = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} Empty_i,\quad
ExplainRate = \frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}} Explain_i
$$

**实现注意**
- 这些指标不依赖 `ground_truth`，只衡量“输出是否落在候选证据内”。  
- 若 $P_i$ 为空，脚本返回 $EC_i=0, HR_i=0, SH_i=0$，并在 `EmptyRate` 中统计。  
- 启用 `--keep-only-cand` 与 `--drop-explain-unless-cand` 后，指标更接近“实体级幻觉”而非“解释文本噪声”。  
- 使用这些指标做论文式对比时建议固定抽取策略（关键词、阈值、是否 containment），否则不同配置下的 $HR/SH$ 不可直接横比。  

### 6.3 消融实验
- hard vs soft
- 不同 top-K 候选数
- 仅实体约束 vs 关系/路径约束

## 7. 失败模式与兜底策略

- **候选集不含真答案**：硬约束会必错；建议使用 soft 或 fallback 机制。
- **实体名多 token**：Trie 必须支持多 token 组合；避免只做单 token allowlist。
- **输出冗长**：prompt 强化 + `--each_line` + 低 `max_new_tokens`。

## 8. 跨项目适配示例（最小集成）

本节给出两个最小适配模板，说明如何把“任意项目的检索输出”映射到 `ConstraintInputs`，不依赖 RA/router。

### 8.1 适配器接口示例（伪代码）
```python
# llm/src/constraints/providers/base.py
from dataclasses import dataclass
from typing import List, Optional, Tuple

Triplet = Tuple[str, str, str]

@dataclass
class ConstraintInputs:
    candidates: List[str]
    evidence_paths: Optional[List[List[Triplet]]] = None
    evidence_relations: Optional[List[str]] = None
    extra_candidates: Optional[List[str]] = None

class ConstraintProvider:
    def build(self, sample: dict) -> ConstraintInputs:
        raise NotImplementedError
```

### 8.2 其他项目的最小适配
假设另一个项目的样本结构如下：
```json
{
  "id": "q1",
  "question": "...",
  "retrieved_entities": ["E1", "E2"],
  "retrieved_triples": [["E1","r1","E3"], ["E2","r2","E4"]]
}
```
对应的 Provider 只需把字段映射到统一协议：
```python
# other_project_adapter.py
from constraints.providers.base import ConstraintProvider, ConstraintInputs

class OtherProjectProvider(ConstraintProvider):
    def build(self, sample: dict) -> ConstraintInputs:
        return ConstraintInputs(
            candidates=sample.get("retrieved_entities", []),
            evidence_paths=[sample.get("retrieved_triples", [])],
        )
```

### 8.3 本项目适配示例（与 GNN-RAG 解耦写法）
```python
# llm/src/constraints/providers/project_adapter.py
from constraints.providers.base import ConstraintProvider, ConstraintInputs

class GnnRagProvider(ConstraintProvider):
    def build(self, sample: dict) -> ConstraintInputs:
        candidates = sample.get("cand", []) or []
        evidence_paths = sample.get("predicted_paths", None)
        return ConstraintInputs(candidates=candidates, evidence_paths=evidence_paths)
```

### 8.4 接入点示例（伪代码）
```python
# predict_answer.py (示意)
provider = GnnRagProvider()
inputs = provider.build(sample)
spec = ConstraintBuilder(tokenizer).build(inputs)
output = model.generate_sentence(prompt, constraints=spec)
```

## 9. 配置建议（即插即用）

新增 CLI 参数（建议）：
```
--constraint_mode {none,entity,relation,path}
--constraint_strength {hard,soft}
--constraint_k 50
--constraint_source {candidates,evidence,union,external}
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

## 10. 里程碑（最小可用版本）

1) 只做实体约束（hard/soft）
2) 集成到 `Llama.generate_sentence()`
3) 新增评估指标（evidence consistency）
4) 扩展到关系/路径约束

## 11. 可执行开发任务清单（按文件拆分）

### 11.1 新增核心模块（约束层）
- `llm/src/constraints/spec.py`
  - 新增 `ConstraintSpec` / `ConstraintInputs` 数据结构（含类型注释与默认值）。
- `llm/src/constraints/trie.py`
  - 实现 Trie 结构：插入 token 序列、按 prefix 返回允许 token 集合。
- `llm/src/constraints/logits.py`
  - `HardConstraintProcessor`：屏蔽不在允许集合内的 token。
  - `SoftConstraintProcessor`：对不在集合内的 token 施加惩罚。
- `llm/src/constraints/entity_vocab.py`
  - 负责实体名规范化（大小写/标点）与多别名融合（可先只支持 canonical name）。
- `llm/src/constraints/builder.py`
  - 从 `ConstraintInputs` 构建 Trie / allowed_tokens；支持 `mode` 切换（entity/relation/path）。

### 11.2 Provider / Adapter 层（解耦输入）
- `llm/src/constraints/providers/base.py`
  - 定义 `ConstraintProvider` 抽象接口。
- `llm/src/constraints/providers/project_adapter.py`
  - 本项目适配：`cand` + `predicted_paths` → `ConstraintInputs`。

### 11.3 LLM 解码接入
- `llm/src/llms/language_models/base_language_model.py`
  - 扩展 `generate_sentence(lm_input, constraints=None)` 签名。
- `llm/src/llms/language_models/llama.py`
  - 保留 pipeline 作为默认路径。
  - 当 `constraints` 非空时切换到 `AutoModelForCausalLM.generate()`，注入 `LogitsProcessorList`。

### 11.4 推理流程接入
- `llm/src/qa_prediction/predict_answer.py`
  - 在 `prediction()` 内调用 Provider 生成 `ConstraintInputs`。
  - 调用 `ConstraintBuilder` 得到 `ConstraintSpec` 并传入 `generate_sentence()`。

### 11.5 Prompt 与输出格式约束（可选但推荐）
- `llm/prompts/llama2_constrained.txt`
  - 新增严格输出模板（仅输出答案列表/每行答案）。
- `llm/src/qa_prediction/build_qa_input.py`
  - 增加可选 `--each_line` 默认推荐设置，降低解析失败率。

### 11.6 可测试性与诊断
- `llm/scripts/eval_hallucination.py`
  - 保留现有指标；在输出结果中加入“fallback 触发次数”等诊断字段（可选）。
- `llm/scripts/eval_hallucination.sh`
  - 新增参数用于固定抽取策略，保证跨实验可比性。

### 11.7 配置与文档同步
- `docs/design_graph_constrained_decoding.md`
  - 随代码调整同步更新模块结构、参数说明与示例。

### 11.8 分阶段实施顺序（先最小可运行，再逐步完善）

**阶段 A：最小可运行版本（MVP）**
- 目标：在不破坏现有推理流程的前提下，实现“实体级硬约束”的最小闭环。
- 任务：
  - `llm/src/constraints/spec.py`：仅实现 `ConstraintSpec`/`ConstraintInputs` 基础字段（entity + hard）。
  - `llm/src/constraints/trie.py`：支持插入 token 序列 + prefix 查询。
  - `llm/src/constraints/logits.py`：实现 `HardConstraintProcessor`。
  - `llm/src/constraints/builder.py`：仅支持 `mode=entity`，输入 `candidates` → Trie。
  - `llm/src/constraints/providers/project_adapter.py`：把 `data["cand"]` 映射为 `ConstraintInputs`。
  - `llm/src/llms/language_models/llama.py`：新增 constrained 分支（`AutoModelForCausalLM.generate()` + `LogitsProcessorList`）。
  - `llm/src/qa_prediction/predict_answer.py`：将 `ConstraintSpec` 传入 `generate_sentence()`。
- 验收：能用 `--constraint_mode entity --constraint_strength hard` 生成不出候选外实体；与无约束跑通同一流程。

**阶段 B：稳定性增强**
- 目标：减少解析失败与“过约束错杀”。
- 任务：
  - `llm/prompts/llama2_constrained.txt` + `--each_line` 推荐配置。
  - `builder.py` 增加 `constraint_k` 截断、空候选保护。
  - 约束失败时的简单 fallback：允许退回 Level-0 格式约束（或直接禁用）。

**阶段 C：软约束与诊断**
- 目标：支持 soft penalty，并可观察约束影响。
- 任务：
  - `llm/src/constraints/logits.py`：实现 `SoftConstraintProcessor`。
  - `builder.py`：支持 `strength=soft`、`lambda` 参数。
  - `eval_hallucination.py`：增加“fallback 触发次数”等诊断字段（可选）。

**阶段 D：关系/路径约束与扩展候选**
- 目标：支持 S2/S3 与更细粒度约束。
- 任务：
  - `builder.py`：支持 `mode=relation/path`（从 `evidence_paths` 或 `evidence_relations` 构造）。
  - Provider 增加 `extra_candidates`（可选扩展候选集合）。

**阶段 E：多答案格式与去重（可选）**
- 目标：更稳定的多答案输出与去重。
- 任务：
  - 状态机支持列表格式与去重（`used_entities`）。
  - 评估脚本中增加多答案一致性统计（可选）。

---

如需我进一步把以上阶段拆成具体 PR（每个 PR 的文件清单 + 验收步骤），告诉我你的优先级。