# Plan Detail

## Stage A - MVP (实体级硬约束闭环)
- [x] 约束核心模块（spec/trie/logits/builder）
- [x] Provider 适配器（GnnRagProvider）
- [x] Llama constrained 解码分支接入
- [x] predict_answer 传递 constraints 参数

## Stage B - 稳定性增强
- [ ] 约束失败 fallback（退回 Level-0 或禁用）
- [ ] candidate 截断与空候选保护完善
- [ ] 约束输出模板（constrained prompt）

## Stage C - 软约束与诊断
- [ ] SoftConstraintProcessor
- [ ] 软约束参数支持（lambda）
- [ ] 诊断输出（fallback 次数等）

## Stage D - 关系/路径约束与扩展候选
- [ ] relation/path mode
- [ ] extra_candidates 扩展

## Stage E - 多答案格式与去重
- [ ] 列表格式状态机
- [ ] 去重与多答案一致性统计
