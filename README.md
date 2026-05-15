# Heuristic Anchors: Overcoming the Retrieval Bottleneck in Length-Constrained Reasoning

This repository contains the official implementation of **EHC**, a student-teacher framework for improving mathematical reasoning in large language models through iterative hint-guided correction and selective reinforcement training.

## Overview

EHC implements a closed-loop learning pipeline where:

1. A **student model** (e.g., DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, DeepSeek-R1-Distill-Qwen-32B, or any other models) attempts math problems
2. A **teacher model** (e.g., API-based) grades answers, identifies mistakes, and generates pedagogical hints
3. The student re-attempts problems using hints, producing **advantageous** (helpful) and **disadvantageous** (unhelpful) hint-answer pairs
4. An **IRDCL dataset** is constructed from these pairs along with correctly-answered anchor data
5. The student is fine-tuned using **SIRA training** — a KL-regularized objective that learns from hints while preserving existing knowledge

## Pipeline Workflow

The full training pipeline is orchestrated through [`main.py`](main.py). Below is the step-by-step workflow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: Student Takes Exam                                             │
│  student_take_exam_Math_sub(train=True)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 2: Teacher Grades Exam                                            │
│  teacher.teacher_mark_paper_with_save()                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 3: Roll-8 Recheck on Mistakes                                     │
│  exam_roll_recheck_mistake() → saves solved to corr_answer.json         │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 4: Verify Correct Answers                                         │
│  teacher.check_answers_equivalence()                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 5: Teacher Generates Hints                                        │
│  teacher.teacher_hints()  (API-based, or use pre-generated hints)       │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 6: Student Corrects with Hints                                    │
│  student_correct()                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 7: Roll-8 Recheck on Hint-based Answers                           │
│  exam_roll_recheck_hints()                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 8: Generate IRDCL Training Dataset                                │
│  gen_IRDCL_dataset_v2(batch_size=4, split=0.75, epoch=10)               │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 9: SIRA Training                                                  │
│  run_sira_training_v3(model_path=model_path, real_data_epochs=10)       │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 10: Evaluation                                                    │
│  student_take_exam_*(lora_path=lora_path)                               │
│  teacher.teacher_mark_paper_with_save()                                 │
│  exam_roll_recheck_mistake(use_lora=True, lora_path=lora_path)          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Detailed Step Descriptions

#### Step 1: Student Takes Exam

The student model generates solutions for the MATH training set using vLLM for efficient batched inference. Results are saved to `datasets/exam/exam.json`.

```python
student_take_exam_Math_sub(train=True)
```

#### Step 2: Teacher Grades Exam

The teacher evaluates each student answer against the reference solution, classifying responses as correct or incorrect. Results are split into `corr_answer.json` and `mistake_collection_book.json`.

```python
teacher = TeacherCorrecter()
teacher.teacher_mark_paper_with_save()
```

#### Step 3: Roll-8 Recheck on Mistakes

For questions the student initially got wrong, we sample 8 additional attempts (temperature=0.7). Questions solved in any of the 8 rolls are moved from the mistake book to `corr_answer.json`, as they represent knowledge the student possesses but failed to demonstrate on the first try.

```python
exam_roll_recheck_mistake()
```

#### Step 4: Verify Correct Answers

Uses the teacher model to verify answer equivalence between student answers and reference answers, handling cases where formatting differences might cause false negatives.

```python
teacher.check_answers_equivalence()
```

#### Step 5: Teacher Generates Hints

The teacher generates pedagogical hints for each remaining mistake. Hints guide the student toward the correct solution without revealing the answer directly. Pre-generated hints (e.g., `hints_DS_MATH_2048_32b.json`) can be used by renaming to `hints.json`.

```python
teacher.teacher_hints()
# Or use pre-generated hints from datasets/exam/
```

#### Step 6: Student Corrects with Hints

The student re-attempts each mistake using the teacher's hints. Responses are classified into:

- **Advantageous hints** (`adv_hints.json`): Hints that helped the student answer correctly
- **Disadvantageous hints** (`disadv_hints.json`): Hints that did not lead to a correct answer

```python
student_correct()
```

#### Step 7: Roll-8 Recheck on Hint-based Answers

Similar to Step 3, performs roll-8 sampling on hint-augmented questions to identify additional advantageous hint-answer pairs.

```python
exam_roll_recheck_hints()
```

#### Step 8: Generate IRDCL Dataset

Constructs the **Interleaved Reinforcement Data for Corrective Learning** dataset by combining:

- Anchor data from `corr_answer.json` — produced by `exam_roll_recheck_mistake()` in Step 3 (knowledge preservation)
- Advantageous hint-answer pairs from `adv_hints.json` — produced by `student_correct()` and `exam_roll_recheck_hints()` in Steps 6-7 (new knowledge acquisition)
- Disadvantageous hint-answer pairs from `disadv_hints.json` (negative signal)

**Prerequisites**: Ensure `adv_hints.json` has been generated by the preceding pipeline steps, and `corr_answer.json` contains the results from `exam_roll_recheck_mistake()`.

The dataset is interleaved with a configurable split ratio and batch size.

```python
gen_IRDCL_dataset_v2(batch_size=4, split=0.75, epoch=10)
```

#### Step 9: SIRA Training

Trains the student model using the **Selective Interleaved Reinforcement with Anchoring** objective:

- **Hint tokens**: Forward KL divergence (KL(ref || student)) + Cross-Entropy loss
- **Answer tokens**: Reverse KL divergence (KL(student || ref)) for knowledge preservation
- **Anchor tokens**: Reverse KL on correctly-answered questions to prevent catastrophic forgetting
- Top-ρ entropy filtering selects the most informative tokens for KL computation

```python
run_sira_training_v3(model_path=model_path, real_data_epochs=10)
```

#### Step 10: Evaluation

Evaluate the trained model (with LoRA adapter) on held-out benchmarks. Each evaluation requires grading via `teacher.teacher_mark_paper_with_save()` after the exam, and optionally `exam_roll_recheck_mistake()` with the LoRA path for pass@8 accuracy:

```python
lora_path = "<output_dir>/checkpoint-target-reached-epoch-N"

# Evaluate on AIME 1983-2024
student_take_exam_AIME_1983_2024(lora_path=lora_path, max_token=8192)
teacher.teacher_mark_paper_with_save()
exam_roll_recheck_mistake(use_lora=True, lora_path=lora_path, max_token=8192)

# Evaluate on GSM8K
student_take_exam_Gsm8k(train=False, lora_path=lora_path)
teacher.teacher_mark_paper_with_save()
exam_roll_recheck_mistake(use_lora=True, lora_path=lora_path)

# Evaluate on MATH (test)
student_take_exam_Math_sub(train=False, lora_path=lora_path)
teacher.teacher_mark_paper_with_save()
exam_roll_recheck_mistake(use_lora=True, lora_path=lora_path)
```

## Project Structure

```
EHC/
├── main.py                          # Main pipeline orchestration
├── configs/
│   ├── celpo_train.yaml             # Training configuration
│   ├── inference_config.yaml        # Inference configuration
│   └── train_config/                # Detailed training configs
├── scripts/
│   ├── inference/
│   │   ├── take_exam.py             # Student inference (vLLM-based)
│   │   └── teacher_correct.py       # Teacher grading & hint generation
│   └── train/
│       ├── student_train_v3.py      # SIRA training implementation
│       ├── sft_train_baseline.py    # SFT baseline
│       ├── sdpo_baseline.py         # SDPO baseline
│       └── student_grpo.py          # GRPO training
├── utils/
│   ├── IO_utils.py                  # File I/O and path management
│   ├── data_utils.py                # IRDCL dataset generation
│   └── model_utils.py               # LoRA merging utilities
├── data_math/                       # Dataset loaders
│   ├── MATH_Sub_data_util.py        # MATH subset loader
│   ├── MATH_500_data_util.py        # MATH-500 loader
│   ├── GSM8K_data_util.py           # GSM8K loader
│   ├── AIME_data_util.py            # AIME loader
│   ├── AIME_1983_2024.py            # AIME 1983-2024 loader
│   └── LiveMath_data_util.py        # LiveMathBench loader
├── datasets/
│   ├── exam/                        # Runtime data (exam results, hints, etc.)
│   ├── data/                        # Cached benchmark datasets
│   └── cache/                       # HuggingFace dataset cache
├── prompt/
│   └── prompts.py                   # Prompt templates
├── metric/
│   └── reward.py                    # Answer verification metrics
├── loggers/                         # Training loggers
├── requirements.txt                 # Python dependencies
└── environment.yml                  # Conda environment specification
```

## Key Data Files

| File                                         | Description                                |
| -------------------------------------------- | ------------------------------------------ |
| `datasets/exam/exam.json`                    | Student exam responses                     |
| `datasets/exam/mistake_collection_book.json` | Questions the student answered incorrectly |
| `datasets/exam/corr_answer.json`             | Verified correct answers (anchor data)     |
| `datasets/exam/hints.json`                   | Teacher-generated hints for mistakes       |
| `datasets/exam/adv_hints.json`               | Advantageous hint-answer pairs             |
| `datasets/exam/disadv_hints.json`            | Disadvantageous hint-answer pairs          |
| `datasets/exam/irdcl_data.json`              | Final IRDCL training dataset               |

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.x
- 1-4 GPUs (A100 80GB recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd EHC

# Create conda environment
conda env create -f environment.yml
conda activate env_name

# Or install via pip
pip install -r requirements.txt
```

### Key Dependencies

- `transformers` — Model loading and tokenization
- `peft` — LoRA adapter training
- `vllm` — High-throughput inference
- `accelerate` — Multi-GPU training
- `datasets` — HuggingFace dataset loading

## Configuration

### Model Path

Set the base model path in [`main.py`](main.py):

```python
model_path = "/path/to/DeepSeek-R1-Distill-Qwen-7B"
```

### Teacher API (for hint generation)

Configure the teacher API endpoint in [`scripts/inference/teacher_correct.py`](scripts/inference/teacher_correct.py):

```python
base_url = "<your-api-base-url>"
api_key = "<your-api-key>"
```

Alternatively, use a local teacher model via vLLM:

```python
teacher.teacher_hints_self(model_path="/path/to/teacher-model")
```

### Training Hyperparameters

Key SIRA training parameters in [`scripts/train/student_train_v3.py`](scripts/train/student_train_v3.py):

| Parameter              | Default | Description                         |
| ---------------------- | ------- | ----------------------------------- |
| `hint_kl_beta`         | 0.3     | Forward KL weight for hint tokens   |
| `answer_kl_beta`       | 1.0     | Reverse KL weight for answer tokens |
| `anchor_kl_beta`       | 1.0     | Reverse KL weight for anchor tokens |
| `top_entropy_quantile` | 0.2     | Top-ρ entropy filtering ratio       |
| `split_r`              | 0.5     | Anchor/hint data split ratio        |

## Usage

### Full Pipeline

```bash
# Run with single GPU
CUDA_VISIBLE_DEVICES=0 python main.py

# Run with multiple GPUs (for vLLM inference)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

### Using Pre-generated Hints

If you have pre-generated hints (e.g., from a 32B teacher model), you can skip the hint generation step:

1. Copy your hints file to `datasets/exam/hints.json`
2. Similarly, if you have pre-computed mistake data, rename it to `corr_answer.json`
3. Proceed directly from Step 6 (student_correct)

### Individual Steps

Each pipeline step can be run independently by uncommenting the relevant lines in the `__main__` block of [`main.py`](main.py).

## Evaluation Benchmarks

| Benchmark      | Function                                  | Description             |
| -------------- | ----------------------------------------- | ----------------------- |
| MATH (test)    | `student_take_exam_Math_sub(train=False)` | Full MATH test set      |
| MATH-500       | `student_take_exam_Math_500()`            | 500-problem MATH subset |
| GSM8K          | `student_take_exam_Gsm8k(train=False)`    | Grade school math       |
| AIME 2024      | `student_take_exam_AIME(year=2024)`       | AMC/AIME competition    |
| AIME 1983-2024 | `student_take_exam_AIME_1983_2024()`      | Full AIME history       |
| LiveMathBench  | `student_take_exam_LiveMath()`            | Live math benchmark     |

## License

This project is released for academic research purposes.

version 2.0
