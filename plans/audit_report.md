# CELPO 项目深度审计报告

## 一、推理与训练输入格式不一致（BUG 级别）

### BUG 1：System Prompt 不一致 ⚠️ 严重

| 阶段 | System Prompt | 文件 |
|------|--------------|------|
| 推理（TakeExam） | `"Please reason step by step and put your final answer within \boxed{}."` | `scripts/inference/take_exam.py:27` |
| SFT 训练（sft_train） | **无 System Prompt** — 只用了 `user` + `assistant` | `scripts/train/sft_train.py:69` |
| SIRA 训练（student_train） | `"Please reason step by step and put your final answer within \boxed{}."` ✅ | `scripts/train/student_train.py:43` |
| GRPO 训练（student_grpo） | **无 System Prompt** — 只用了 `user` role | `scripts/train/student_grpo.py:81` |
| CELPO 训练（celpo_train） | **无 System Prompt** — 直接用 `prompt` 字段 | `celpo/celpo_train.py:199` |

**影响**：`sft_train.py` 和 `student_grpo.py` 训练时没有 System Prompt，但推理时有。这意味着模型在训练和推理时看到的 token 序列不同，会导致：
- SFT 学到的分布与推理时的条件分布不匹配
- GRPO 的 reward signal 在推理时无法复现

### BUG 2：IRDCL 数据集 `type` 字段不匹配 ⚠️ 中等

`generate_irdcl_dataset()` 在 `data_utils.py:503` 中将 hint 数据标记为 `"type": "hint_data"`，但 `FixedModeCollator` 在 `student_train.py:171` 中检查的是：

```python
data_type = item.get('type', 'anchor_data')
# 只有 data_type != 'anchor_data' 时才走 Mode B 分支
```

而 `generate_irdcl_datase_v2()` 在 `data_utils.py:323` 中标记为 `"type": "mode_b_generation"`。

**影响**：如果使用 `generate_irdcl_dataset()`（v1）生成的数据，`type` 为 `"hint_data"` 而非 `"anchor_data"`，虽然能进入 Mode B 分支（因为不等于 `anchor_data`），但这是偶然正确，不是设计意图。如果未来有人改了默认值就会出 bug。

### BUG 3：Anchor 数据的 `answer` 字段含义混乱 ⚠️ 中等

在 IRDCL 数据集生成中：
- **Hint 数据**的 `answer` 字段 = `item.get("student_answer")` — 这是学生的答案（`data_utils.py:498`）
- **Anchor 数据**的 `answer` 字段 = `item.get("answer")` — 这是 `corr_answer.json` 中的学生答案

但在 `FixedModeCollator` 中，`answer` 被直接用作训练 target：
```python
c = item.get('answer')  # student_train.py:170
```

**问题**：对于 Mode B（hint_data），训练 target 是学生的答案而非参考答案（`ref_solution`）。如果学生的答案格式不规范（比如缺少 `\boxed{}`），模型会学到错误的输出格式。Anchor 数据同理 — 用的是学生答案而非标准答案。

**建议**：Anchor 数据应该用 `ref_solution` 作为训练 target，而非学生答案。

### BUG 4：`filter_json_by_question_idx` 函数签名与调用不匹配 ⚠️ 低

`data_utils.py:126` 定义为 `filter_json_by_question_idx(exam_path, hints_exam_result_path)`（2 个参数），但 `main.py:337` 调用时传了 3 个参数：
```python
filter_json_by_question_idx(exam_paper.exam_file_path, exam_paper.hints_file_path, exam_paper.corr_path)
```
且函数内部引用了未定义的 `corr_path` 变量（`data_utils.py:150`）。这个函数会直接报错。

---

## 二、精度（dtype）不一致 ⚠️ 严重

| 组件 | dtype | 文件:行 |
|------|-------|---------|
| 推理（vLLM） | `bfloat16` | `take_exam.py:106` |
| SFT 训练 | `bfloat16` | `sft_train.py:101` |
| SIRA v1 训练 | **`float32`** ❌ | `student_train.py:458` |
| SIRA v2 训练 | `bfloat16` ✅ | `student_train_v2.py:530` |
| GRPO 训练 | `bfloat16` | `student_grpo.py:130` |
| CELPO 训练 | **`float16`** ❌ | `celpo_train.py:238,293` |

**关键问题**：

1. **SIRA v1 用 float32 训练**（`student_train.py:458`），且 `fp16=False, bf16=False`（`student_train.py:464`）。这意味着整个训练在 float32 下进行，但推理用 bfloat16。float32 训练的 LoRA 权重在 bfloat16 推理时会有精度损失，可能导致训练效果无法完全复现。

2. **CELPO 用 float16 而非 bfloat16**（`celpo_train.py:238,293`）。float16 的动态范围比 bfloat16 小得多，对于 7B 模型容易出现 overflow/underflow。而且推理用 bfloat16，训练用 float16，数值行为不一致。

3. **LoRA rank 不一致**：
   - SIRA v1/v2: `r=16, lora_alpha=32`，只覆盖 `q_proj, k_proj, v_proj, o_proj`
   - SFT/GRPO: `r=64, lora_alpha=128`，覆盖 `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
   
   如果 GRPO 阶段要加载 SIRA 的 LoRA 权重，rank 和 target_modules 不匹配会导致加载失败或部分参数丢失。

---

## 三、Q → H + A 学习目标是否可行？

### 设计分析

核心思路是让模型学会：给定 Question，先生成 Hint（`# known:\n{hints}\n`），再生成 Answer。这通过 Mode B 实现：

```
Input:  [System] + [User: Question] + [Generation Prompt]
Target: # known:\n{hints}\n# Answer:\n{answer}
```

### 可行性评估

**理论上可行，但存在以下问题**：

1. **Hint 质量瓶颈**：Hint 由外部 Teacher（可能是 API 调用）生成，质量取决于 Teacher 模型。如果 Hint 过于具体（直接给出计算步骤），模型学到的是"复述提示"而非"推理能力"。从 `TEACHER_CORRECT_PROMPT`（`prompts.py:100-143`）来看，prompt 设计得不错，强调了"给工具不给答案"，但实际效果取决于 Teacher 模型的遵循度。

2. **Hint 格式与推理格式冲突** ⚠️：
   - 训练时模型学习生成 `# known:\n{hints}\n# Answer:\n{answer}` 格式
   - 但推理时（`take_exam.py:247-254`），System Prompt 要求 `\boxed{}` 格式
   - 模型在推理时不会自动生成 `# known:` 前缀，因为推理 prompt 没有引导它这样做
   - **这意味着 Mode B 学到的 Hint 生成能力在推理时无法被激活**

3. **student_answer vs ref_solution 作为 target**：如前所述，Mode B 的 answer 部分用的是 `student_answer`（学生之前答对的答案），而非 `ref_solution`。学生答案可能格式不规范、推理路径冗长，不是最优的学习目标。

### 核心矛盾

训练目标是让模型学会 `Q → H + A`，但推理时的 prompt 格式是 `Q → \boxed{A}`。模型在推理时没有机会展示它学到的 Hint 生成能力。除非推理时也改为先生成 Hint 再生成 Answer 的格式，否则 Mode B 的训练效果主要通过"隐式正则化"间接影响模型，而非直接使用。

**建议**：
- 推理时也采用两阶段格式：先让模型生成 `# known:` 部分，再生成答案
- 或者将 Mode B 的目标改为 Chain-of-Thought 格式，与推理时的 `\boxed{}` 格式兼容

---

## 四、防遗忘机制分析

### 现有机制

1. **Anchor Data（Mode A）**：在 IRDCL 数据集中混入模型已经答对的题目（`corr_answer.json`），作为 SFT anchor，防止模型在学习新知识时遗忘已有能力。

2. **Reference Model 约束**：`SequentialTrainer.compute_loss()` 中通过 `self.model.disable_adapter()` 获取 reference model 的 logits，用于计算 anchor loss 的动态权重（`student_train.py:234-237`）。当 anchor loss 超过 reference model 的 loss 时，通过 sigmoid 抑制机制降低 anchor 的梯度贡献。

3. **Task Reweighting**：Mode A 和 Mode B 各占 50% 的 loss 权重（`student_train.py:356-358`），确保两个任务的梯度贡献平衡。

4. **Norm-based Scaling**：用 Mode B 的 L2 norm 与总 L2 norm 的比值来缩放最终 loss（`student_train.py:373-384`），倾向于让 Mode B 主导梯度方向。

### 问题

1. **Anchor 数据量可能不足** ⚠️：`corr_answer.json` 中的数据是模型在特定数据集上答对的题目。如果训练数据集很大（50 个 epoch × hints 数据），anchor 数据会被反复重复使用（`data_utils.py:346-351` 中的 `repeat_factor`），导致 anchor 过拟合。

2. **Norm Scaling 可能导致 Anchor 梯度消失** ⚠️：`student_train.py:378` 中 `scale_factor = norm_b / norm_total`。如果 Mode B loss 很小（模型已经学会了 Hint），`norm_b` 接近 0，整个 loss 会被缩放到接近 0，包括 anchor 部分。这会导致模型停止学习 anchor 数据，加速遗忘。

3. **没有 Replay Buffer 或 EWC**：当前的防遗忘机制完全依赖 anchor data mixing。没有使用更强的持续学习技术（如   ）。对于多轮迭代训练，这可能不够。

4. **GRPO 阶段无防遗忘机制**：`student_grpo.py` 中的 GRPO 训练没有任何 anchor 数据或 KL 约束（虽然有 `beta=0.04` 的 KL penalty，但这是相对于 GRPO 自身的 reference policy，不是相对于原始模型）。如果在 SIRA 之后再跑 GRPO，可能会覆盖 SIRA 学到的能力。

---

## 五、训练参数审计

### SIRA v1（`student_train.py:445-467`）

| 参数 | 值 | 评估 |
|------|-----|------|
| learning_rate | `1e-4` | ⚠️ 偏高。对于 7B 模型的 LoRA 微调，通常用 `1e-5` ~ `5e-5` |
| warmup_ratio | `0` | ⚠️ 无 warmup，配合高 lr 容易导致训练初期不稳定 |
| lr_scheduler | 默认 linear | OK |
| batch_size | `2 * gradient_accum` = 有效 16 | OK |
| dtype | `float32` | ⚠️ 浪费显存，且与推理精度不一致 |
| LoRA r | 16 | OK，但与 SFT/GRPO 的 r=64 不一致 |
| epochs | 20 | 取决于数据量 |

### SIRA v2（`student_train_v2.py:466-563`）

| 参数 | 值 | 评估 |
|------|-----|------|
| learning_rate | `3e-4` | ⚠️ 非常高。即使是 LoRA，3e-4 对 7B 模型也偏激进 |
| warmup_ratio | `0.05` | ✅ 有 warmup |
| lr_scheduler | `cosine` | ✅ |
| dtype | `bfloat16` | ✅ |
| LoRA r | 16 | OK |
| anchor_sigmoid_slope | `50.0`（v2）vs `1000.0`（v1） | v2 更合理，v1 的 1000 基本等于 hard threshold |
| suppress_max_scale | `1.0`（v2）vs `5.0`（v1） | v2 更保守 |

### GRPO（`student_grpo.py:112-139`）

| 参数 | 值 | 评估 |
|------|-----|------|
| learning_rate | `1e-6` | ✅ RL 阶段用低 lr 是正确的 |
| num_generations | 8 | ✅ |
| max_completion_length | 1024 | ⚠️ 数学推理可能需要更长（推理时用 2048） |
| beta (KL) | `0.04` | ✅ 合理范围 |
| num_train_epochs | 1 | ✅ RL 通常只跑 1 epoch |

### CELPO（`celpo/celpo_train.py:278-308`）

| 参数 | 值 | 评估 |
|------|-----|------|
| learning_rate | `1e-5` | ✅ |
| fp16 | `True` | ⚠️ 应该用 bf16（见精度分析） |
| per_device_batch_size | 1 | ⚠️ 太小，梯度噪声大 |
| gradient_accumulation | 8 | 有效 batch = 8，偏小 |
| num_generations | 4 | ⚠️ 偏少，GRPO 通常需要 8-16 |
| beta | `0.01` | ⚠️ 偏小，可能导致 policy 偏离过大 |

---

## 六、总结：关键修复优先级

### P0（必须修复，影响正确性）

1. **统一 System Prompt**：所有训练脚本（`sft_train.py`, `student_grpo.py`）必须加上与推理一致的 System Prompt
2. **修复 `filter_json_by_question_idx` 函数签名**：参数数量和内部变量引用不匹配
3. **统一 dtype 为 bfloat16**：`student_train.py` 的 float32 和 `celpo_train.py` 的 float16 都应改为 bfloat16

### P1（强烈建议修复，影响效果）

4. **训练 target 应使用 `ref_solution` 而非 `student_answer`**：至少对 Anchor 数据如此
5. **降低 SIRA v1 的 learning_rate**：从 `1e-4` 降到 `2e-5`，并加 warmup
6. **降低 SIRA v2 的 learning_rate**：从 `3e-4` 降到 `5e-5`
7. **统一 LoRA 配置**：确保 SIRA 和 GRPO 使用相同的 rank 和 target_modules，否则权重无法正确传递

### P2（建议优化，提升效果）

8. **解决 Mode B 训练与推理格式不兼容的问题**：推理时也应支持 Hint 生成格式，或将 Mode B 改为 CoT 格式
9. **修复 Norm Scaling 导致的梯度消失风险**：当 Mode B loss 很小时，应有下限保护
10. **GRPO 的 `max_completion_length` 应与推理一致**：从 1024 提升到 2048
11. **CELPO 训练的 `num_generations` 应增加到 8**
