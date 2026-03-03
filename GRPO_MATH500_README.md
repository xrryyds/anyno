# GRPO on MATH500 完整指南

本文档说明如何使用 SIRA 训练结果在 MATH500 数据集上进行 GRPO 训练，并测试模型性能。

## 📋 目录

1. [工作流程概述](#工作流程概述)
2. [前置要求](#前置要求)
3. [步骤详解](#步骤详解)
4. [代码说明](#代码说明)
5. [常见问题](#常见问题)

---

## 🔄 工作流程概述

完整的训练和测试流程包含三个步骤：

```
Step 1: SIRA 训练
   ↓
   生成 checkpoint (LoRA 权重)
   ↓
Step 2: GRPO 训练
   ↓
   生成 GRPO checkpoint
   ↓
Step 3: 测试评估
   ↓
   输出准确率报告
```

---

## ✅ 前置要求

### 1. 数据准备

确保以下数据文件存在：
- `datasets/exam/irdcl_data.json` - SIRA 训练数据
- MATH500 数据集会自动加载

### 2. 环境配置

```bash
# 安装依赖
pip install transformers peft trl datasets torch

# 设置环境变量（如需要）
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 3. 模型路径

在 [`main.py`](main.py:44) 中配置基础模型路径：
```python
model_path = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"
```

---

## 📝 步骤详解

### Step 1: SIRA 训练

**目的**: 使用 SIRA 方法训练模型，学习如何利用 hints 提升推理能力

**运行方式**:
```bash
# 方式 1: 使用示例脚本
python example_grpo_workflow.py --step 1

# 方式 2: 直接在 main.py 中调用
python -c "from scripts import run_sira_training_v2; run_sira_training_v2(model_path='/path/to/model', target_mode_b=0.023)"
```

**关键参数**:
- `batch_size`: 批次大小，默认 8
- `real_data_epochs`: 最大训练轮数，默认 50
- `target_mode_b`: 目标 Mode B Loss，达到后自动停止（默认 0.023）

**输出**:
- Checkpoint 保存在: `output/sira_sft_50ep_MMDD_HHMM/checkpoint-target-reached-epoch-X/`
- 训练日志: `output/sira_sft_50ep_MMDD_HHMM/train.log`
- 指标记录: `output/sira_sft_50ep_MMDD_HHMM/epoch_metrics.jsonl`

**训练监控**:
根据您的训练结果 ([`epoch_metrics.jsonl`](output/sira_sft_50ep_0302_1046/epoch_metrics.jsonl:8))，模型在第 8 轮达到了很好的效果：
```json
{
  "avg_mode_b_loss_raw": 0.010607,
  "avg_anchor_loss_raw": 0.024992,
  "logical_epoch": 8
}
```

---

### Step 2: GRPO 训练

**目的**: 使用 Group Relative Policy Optimization 在 MATH500 上进一步优化模型

**运行方式**:
```bash
# 方式 1: 使用示例脚本
python example_grpo_workflow.py --step 2 \
    --sira-checkpoint /path/to/sira/checkpoint

# 方式 2: 多卡训练（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    example_grpo_workflow.py --step 2 \
    --sira-checkpoint /root/autodl-tmp/CELPO/output/sira_sft_50ep_0302_1046/checkpoint-target-reached-epoch-10

# 方式 3: 在 main.py 中调用
python main.py  # 取消注释 grpo_on_MATH500() 调用
```

**关键参数**:
- `lora_path`: SIRA 训练的 checkpoint 路径
- `num_generations`: 每个问题生成的样本数（默认 8）

**GRPO 配置** (在 [`student_grpo.py`](scripts/train/student_grpo.py:112)):
```python
GRPOConfig(
    learning_rate=1e-6,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_generations=8,
    beta=0.04,
    num_train_epochs=1
)
```

**输出**:
- 最终模型保存在: `output/grpo_stable/`

---

### Step 3: 测试评估

**目的**: 在 MATH500 测试集上评估 GRPO 训练后的模型性能

**运行方式**:
```bash
# 方式 1: 使用示例脚本
python example_grpo_workflow.py --step 3 \
    --grpo-checkpoint /root/autodl-tmp/CELPO/output/grpo_stable

# 方式 2: 在 main.py 中调用
python -c "from main import test_grpo_on_MATH500; test_grpo_on_MATH500('/path/to/grpo/checkpoint')"
```

**测试流程**:
1. 加载 MATH500 数据集（500 道题）
2. 使用 GRPO LoRA 进行推理
3. 自动批改答案
4. 输出统计报告

**输出示例**:
```
============================================================
📊 GRPO MODEL PERFORMANCE ON MATH500
============================================================
Total Questions    : 500
Correct Answers    : 387
Incorrect Answers  : 113
Accuracy           : 77.40%
============================================================
```

---

## 💻 代码说明

### 1. [`grpo_on_MATH500()`](main.py:671) 函数

完善后的函数包含：
- 详细的日志输出
- 参数验证
- 数据集加载
- GRPO 训练调用

```python
def grpo_on_MATH500(lora_path: str, num_generations: int = 8):
    """在 MATH500 数据集上进行 GRPO 训练"""
    logger.info("Starting GRPO Training on MATH500")
    
    data = Math_500()
    question = data.problems
    answer = data.answers
    
    run_grpo_training(
        base_model_path=model_path, 
        sft_lora_path=lora_path, 
        questions=question, 
        answers=answer,
        num_generations=num_generations
    )
```

### 2. [`test_grpo_on_MATH500()`](main.py:786) 函数

新增的测试函数包含：
- 模型加载和推理
- 自动批改
- 统计分析
- 结果输出

```python
def test_grpo_on_MATH500(grpo_lora_path: str):
    """测试 GRPO 训练后的模型在 MATH500 上的表现"""
    data = Math_500()
    
    # 使用 GRPO LoRA 进行推理
    take_exam = TakeExam(
        model_path=model_path,
        use_lora=True,
        adapter_path=grpo_lora_path
    )
    take_exam.exam(question, solution, answer, question_idx)
    
    # 批改并统计
    teacher = TeacherCorrecter()
    incorrect_data, correct_data = teacher.teacher_mark_paper()
    
    # 返回结果
    return {
        "total": total_count,
        "correct": num_correct,
        "accuracy": accuracy
    }
```

### 3. [`example_grpo_workflow.py`](example_grpo_workflow.py:1)

完整的工作流脚本，支持：
- 命令行参数控制
- 分步执行
- 路径配置
- 错误处理

---

## ❓ 常见问题

### Q1: SIRA 训练需要多长时间？

根据您的配置和硬件：
- 单卡 (8GB VRAM): 约 2-3 小时
- 多卡 (4x8GB): 约 30-60 分钟
- 早停机制会在达到目标 loss 时自动停止

### Q2: GRPO 训练显存不足怎么办？

调整 [`student_grpo.py`](scripts/train/student_grpo.py:119) 中的参数：
```python
per_device_train_batch_size=4,  # 降低批次大小
gradient_accumulation_steps=8,  # 增加累积步数
```

### Q3: 如何选择最佳的 SIRA checkpoint？

查看 `epoch_metrics.jsonl`，选择 `avg_mode_b_loss_raw` 最低的 epoch：
```bash
cat output/sira_sft_50ep_*/epoch_metrics.jsonl | jq '.avg_mode_b_loss_raw'
```

### Q4: 测试时出现 OOM 错误？

在 [`take_exam.py`](scripts/inference/take_exam.py:101) 中降低 `gpu_memory_utilization`：
```python
gpu_memory_utilization=0.7,  # 从 0.9 降低到 0.7
```

### Q5: 如何对比 GRPO 前后的性能？

```bash
# 测试 SIRA 模型
python main.py  # 调用 student_take_exam_Math_500(lora_path="sira_checkpoint")

# 测试 GRPO 模型
python example_grpo_workflow.py --step 3 --grpo-checkpoint "grpo_checkpoint"
```

---

## 📊 性能基准

根据您的训练结果：

| 阶段 | Mode B Loss | Anchor Loss | 说明 |
|------|-------------|-------------|------|
| Epoch 1 | 0.2109 | 0.0238 | 初始状态 |
| Epoch 5 | 0.0809 | 0.0251 | 快速下降 |
| Epoch 8 | 0.0106 | 0.0250 | 接近收敛 |

**建议**: 使用 Epoch 8-10 的 checkpoint 进行 GRPO 训练

---

## 🚀 快速开始

```bash
# 1. SIRA 训练
python example_grpo_workflow.py --step 1

# 2. 等待训练完成，记录 checkpoint 路径
# 例如: output/sira_sft_50ep_0302_1046/checkpoint-target-reached-epoch-10

# 3. GRPO 训练（多卡推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    example_grpo_workflow.py --step 2 \
    --sira-checkpoint output/sira_sft_50ep_0302_1046/checkpoint-target-reached-epoch-10

# 4. 测试模型
python example_grpo_workflow.py --step 3 \
    --grpo-checkpoint output/grpo_stable
```

---

## 📚 相关文件

- [`main.py`](main.py:671) - 主函数定义
- [`student_train_v2.py`](scripts/train/student_train_v2.py:465) - SIRA 训练实现
- [`student_grpo.py`](scripts/train/student_grpo.py:52) - GRPO 训练实现
- [`take_exam.py`](scripts/inference/take_exam.py:48) - 推理引擎
- [`example_grpo_workflow.py`](example_grpo_workflow.py:1) - 工作流脚本

---

## 📝 更新日志

- 2026-03-02: 完善 `grpo_on_MATH500()` 函数
- 2026-03-02: 新增 `test_grpo_on_MATH500()` 测试函数
- 2026-03-02: 创建完整工作流示例脚本
