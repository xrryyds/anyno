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
| learning_rate | `1e-5` | ✅ 合理范围 |
| warmup_ratio | `0.05` | ✅ 有 warmup |
| lr_scheduler | `cosine` | ✅ |
| dtype | `bfloat16` | ✅ |
| LoRA r | 16 | OK |
| anchor_sigmoid_slope | `50.0`（v2）vs `1000.0`（v1） | v2 更合理，v1 的 1000 基本等于 hard threshold |
| suppress_max_scale | `1.0`（v2）vs `5.0`（v1） | v2 更保守 |
| anchor_loss_tolerance | `1.01` | ✅ 非常紧的容忍度，几乎不允许 anchor loss 上升 |

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

## 六、训练数据量与有效性分析（14 adv_hints + r=0.875）

### 6.1 实际数据统计

根据对当前数据文件的分析：

| 文件 | 总条目 | 唯一题目数 | 每题答案数 |
|------|--------|-----------|-----------|
| `adv_hints.json` | 70 | 14 | 5（每题 Roll@8 中 5 条正确） |
| `corr_answer.json`（当前） | 42 | 12 | 1~5 不等 |
| `irdcl_data.json` | 5600 | — | — |
| ├─ hint_data | 700 | 14 | 每题重复 50 次 |
| └─ anchor_data | 4900 | 447 | 每题重复 ~11 次 |

> **注意**：当前 `corr_answer.json` 只有 12 个唯一题目（42 条），但 IRDCL 中的 anchor 数据有 447 个唯一题目。这说明 IRDCL 是用**之前的** corr 文件（包含 Pass@1 答对的 ~447 题）生成的，之后 `corr_answer.json` 被 [`exam_roll_recheck_mistake()`](main.py:487) 重写为只包含 Roll@K 新解出的 12 题。**当前 `corr_answer.json` 缺少 `ref_beta` 字段**，如果重新生成 IRDCL 会出错。

### 6.2 IRDCL 数据集构成分析

调用参数：[`gen_IRDCL_dataset(8, 0.875, 10)`](main.py:467)

- `batch_size=8`, `anchor_k=0.875` → `std_num_anchors = int(8 * 0.875) = 7`, `std_num_hints = 1`
- `epoch=10`
- 每个 epoch：70 条 hint / 1 条 per batch = 70 个 batch，每 batch 配 7 条 anchor
- 10 个 epoch → 70 × 10 = **700 hint items** + 70 × 7 × 10 = **4900 anchor items** = **5600 total**

**每条 hint 数据被看到的次数**：

```
70 条 hint × 10 epoch = 700 次出现
14 个唯一题目 → 每个题目出现 700/14 = 50 次
```

**每条 anchor 数据被看到的次数**：

```
4900 anchor items / 447 唯一题目 ≈ 11 次/题
```

### 6.3 训练步数计算

在 [`run_sira_training_v2()`](scripts/train/student_train_v2.py:465) 中：

```python
total_samples = 5600
total_steps = 5600 // 8 = 700  # batch_size=8
steps_per_logical_epoch = 700 // 50 = 14  # real_data_epochs=50
```

- **总训练步数**：700 步
- **每个 logical epoch**：14 步
- **50 个 logical epoch** 用于 early stopping 检查

### 6.4 有效性评估

#### ✅ 可行的方面

1. **Anchor 数据覆盖面足够**：447 个唯一题目的 anchor 数据覆盖了原始 500 题中 ~89.4% 的题目，这为防遗忘提供了充分的数据基础。

2. **r=0.875 的比例设计合理**：
   - 每个 batch 中 1 hint + 7 anchor，确保模型在学习新知识时有大量"旧知识复习"
   - [`adaptive_gating_loss()`](scripts/train/student_train_v2.py:282) 中的 task reweighting（line 367-370）将 hint 和 anchor 各分配 50% 权重，所以 1 条 hint 获得 50% 的 loss 权重，7 条 anchor 平分另外 50%
   - 这意味着 hint 的**单条梯度贡献**是 anchor 的 7 倍，有效放大了少量 hint 数据的学习信号

3. **Anchor suppression 机制有效**：
   - `anchor_loss_tolerance=1.01` 意味着只要 anchor loss 超过 ref_beta 的 1.01 倍就会被抑制
   - `anchor_sigmoid_slope=50.0` 提供了较陡的抑制曲线
   - 这确保 anchor 不会过度训练，只在模型"快要遗忘"时才产生有效梯度

4. **Norm-based scaling 倾向 Mode B**：[line 373-379](scripts/train/student_train_v2.py:373) 用 `norm_b / norm_total` 缩放最终 loss，使梯度方向偏向 Mode B（hint 学习），这对少量 hint 数据是有利的。

#### ⚠️ 存在的风险

1. **14 个唯一题目的 hint 数据严重不足**：
   - 14 个题目 × 5 个答案 = 70 条 hint 数据，但只覆盖 14 种"知识模式"
   - 每个题目被看到 50 次（10 epoch × 5 答案），**极易过拟合到这 14 个特定题目的模式**
   - 模型可能学会的是"记住这 14 道题的答案"，而非"学会利用 hint 进行推理"的通用能力
   - **类比**：这相当于让学生反复做同样 14 道题 50 遍，学生会背答案而非理解方法

2. **Hint 多样性不足**：
   - 14 个题目的 hint 来自同一个 Teacher 模型，hint 的风格和知识类型高度相似
   - 如果这 14 题集中在某些数学子领域（如代数），模型不会学到其他领域（如几何、数论）的 hint 利用能力

3. **Gate 机制可能过早饱和**：
   - [`gate_threshold=0.3`](scripts/train/student_train_v2.py:54)，[`gate_slope=3.0`](scripts/train/student_train_v2.py:55)
   - 当 hint loss < 0.3 时，gate → 1.0（完全打开 answer loss）
   - 由于只有 14 个题目重复 50 次，hint loss 会很快降到 0.3 以下
   - 一旦 gate 饱和，Mode B 就退化为普通 SFT（hint + answer 都全权重学习）
   - 这时 gate 机制失去了"先学 hint 再学 answer"的课程学习效果

4. **corr_answer.json 数据不一致** ⚠️ 严重：
   - 当前 `corr_answer.json` 只有 12 题 42 条，且**没有 `ref_beta` 字段**
   - 但 IRDCL 中的 anchor 数据有 447 题且包含 `ref_beta`
   - 这说明 IRDCL 是用旧版 corr 文件生成的，之后 corr 被覆盖
   - 如果需要重新生成 IRDCL，[`gen_IRDCL_dataset()`](main.py:467) 会先调用 [`compute_and_save_ref_loss()`](main.py:422) 计算 ref_beta，但只会为当前 12 题计算
   - **结果**：重新生成的 IRDCL 只有 12 题 anchor，远不足以防遗忘

### 6.5 量化评估：这样训练能学到什么？

| 指标 | 值 | 评估 |
|------|-----|------|
| Hint 题目覆盖率 | 14/500 = 2.8% | ❌ 极低 |
| Anchor 题目覆盖率 | 447/500 = 89.4% | ✅ 充足 |
| Hint 每题重复次数 | 50 次 | ⚠️ 过高，过拟合风险 |
| Anchor 每题重复次数 | ~11 次 | ✅ 合理 |
| 有效训练步数 | 700 步 | ⚠️ 偏少 |
| Hint 梯度放大倍数 | 7× (vs anchor) | ✅ 合理 |

### 6.6 结论与建议

**当前配置的训练是否有效？**

- **防遗忘**：✅ 有效。447 题 anchor + r=0.875 + anchor suppression 机制，足以维持模型在已答对题目上的能力。
- **学习新知识**：⚠️ 效果有限。14 个题目的 hint 数据太少，模型大概率会过拟合到这 14 个特定模式，而非学到通用的 hint 利用能力。

**改进建议**：

1. **增加 hint 数据量**：
   - 使用更大的训练集（如 MATH train 的 7500 题而非 500 题）
   - 或降低 Pass@1 的阈值，让更多"边界题"进入 hint 流程
   - 目标：至少 50-100 个唯一 hint 题目

2. **减少 epoch 数**：
   - 当前 `epoch=10` 导致每题重复 50 次，建议降到 `epoch=3`（每题 15 次）
   - 配合 early stopping（`target_mode_b=0.023`）防止过拟合

3. **增加 hint 数据增强**：
   - 对同一题目生成多个不同风格的 hint（不同 Teacher prompt）
   - 或对 hint 进行随机 dropout（训练时随机删除部分 hint 条目），增加泛化能力

4. **修复 corr_answer.json 数据**：
   - 恢复原始的 Pass@1 正确答案文件（447 题）
   - 将 Roll@K 新解出的 12 题合并进去，而非覆盖
   - 重新运行 [`compute_and_save_ref_loss()`](main.py:422) 为所有 anchor 计算 ref_beta

---

## 七、总结：关键修复优先级

### P0（必须修复，影响正确性）

1. **修复 corr_answer.json 数据丢失**：恢复 447 题的原始 corr 数据，合并 Roll@K 新解出的题目，重新计算 ref_beta
2. **统一 System Prompt**：所有训练脚本（`sft_train.py`, `student_grpo.py`）必须加上与推理一致的 System Prompt
3. **修复 `filter_json_by_question_idx` 函数签名**：参数数量和内部变量引用不匹配
4. **统一 dtype 为 bfloat16**：`student_train.py` 的 float32 和 `celpo_train.py` 的 float16 都应改为 bfloat16

### P1（强烈建议修复，影响效果）

5. **增加 hint 数据量**：14 个唯一题目远不足以学到通用能力，目标至少 50-100 题
6. **减少 epoch 数**：从 10 降到 3，避免 hint 数据过拟合
7. **训练 target 应使用 `ref_solution` 而非 `student_answer`**：至少对 Anchor 数据如此
8. **统一 LoRA 配置**：确保 SIRA 和 GRPO 使用相同的 rank 和 target_modules，否则权重无法正确传递

### P2（建议优化，提升效果）

9. **解决 Mode B 训练与推理格式不兼容的问题**：推理时也应支持 Hint 生成格式，或将 Mode B 改为 CoT 格式
10. **修复 Norm Scaling 导致的梯度消失风险**：当 Mode B loss 很小时，应有下限保护
11. **增加 hint 数据增强**：多风格 hint、hint dropout 等
12. **GRPO 的 `max_completion_length` 应与推理一致**：从 1024 提升到 2048
