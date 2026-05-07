

GRPOGroup Relative Policy Optimization


```
scripts/Inference/
├── inference.py
├── run_inference.py
└── __init__.py

configs/
├── inference_config.yaml
└── ...

outputs/
```


```bash
python scripts/Inference/run_inference.py

python scripts/Inference/run_inference.py --config configs/inference_config.yaml
```


```bash
python scripts/Inference/run_inference.py --question ": x^2 - 5x + 6 = 0"
```


```bash
echo " 1+2+3+...+100 " > input.txt
echo "5" >> input.txt
echo ": 2x + 3 = 7" >> input.txt

python scripts/Inference/run_inference.py --batch-input input.txt --output results.json
```


```bash
python scripts/Inference/run_inference.py --interactive
```


```bash
python scripts/Inference/run_inference.py --evaluate --max-samples 100
```


### inference_config.yaml

```yaml
model:
  model_path: "./outputs/grpo_fix_final_model"
  base_model: "Qwen/Qwen2-1.5B-Instruct"
  use_lora: true
  lora:
    r: 16
    alpha: 32                                   # LoRA Alpha
    dropout: 0.05                               # LoRA Dropout

inference:
  max_length: 1024
  max_new_tokens: 512
  temperature: 0.8
  top_p: 0.9
  top_k: 50
  num_return_sequences: 1
  device: "cuda"

evaluation:
  dataset_name: "HuggingFaceH4/MATH-500"
  split: "test"
  max_samples: 100

output:
  results_path: "./inference_results.json"
  save_detailed_log: true
  log_path: "./inference_log.txt"

optimization:
  batch_size: 4
  gradient_checkpointing: false
  fp16: true
```


```python
from scripts.Inference.inference import GRPOInference, InferenceConfig

config = InferenceConfig(
    model_path="./outputs/grpo_fix_final_model",
    base_model="Qwen/Qwen2-1.5B-Instruct",
    max_new_tokens=512
)

inference = GRPOInference(config)

problem = ": x^2 - 5x + 6 = 0"
prompt = inference.preprocess_prompt(problem)
result = inference.generate_response(prompt)

print(f": {problem}")
print(f": {result['answer']}")
print(f": {result['response']}")
```


```python
problems = [
    " 1+2+3+...+100 ",
    "5",
    ": 2x + 3 = 7"
]

results = inference.batch_inference(problems, batch_size=2)

for result in results:
    print(f": {result['problem']}")
    print(f": {result['answer']}")
```


```python
eval_results = inference.evaluate_on_dataset(
    dataset_name="HuggingFaceH4/MATH-500",
    split="test",
    max_samples=100,
    batch_size=4
)

print(f": {eval_results['accuracy']:.4f}")
print(f": {eval_results['correct']}/{eval_results['total']}")
```


```json
{
  "prompt": "prompt",
  "response": "",
  "answer": "",
  "full_output": ""
}
```


```json
[
  {
    "problem": "",
    "prompt": "prompt",
    "response": "",
    "answer": "",
    "index": 0
  },
  ...
]
```


```json
{
  "accuracy": 0.85,
  "correct": 85,
  "total": 100,
  "eval_time": 120.5,
  "results": [
    {
      "index": 0,
      "problem": "",
      "reference_answer": "",
      "model_answer": "",
      "is_correct": true,
      "model_response": "",
      "reference_answer": ""
    },
    ...
  ]
}
```


-  `batch_size` 
- GPU
- 


- `preprocess_prompt`  LRU 
- prompt
- 1000


- GPU
- 
- 


1. 
2. 
3. LoRA

```bash
ls -la ./outputs/grpo_fix_final_model/
```


1.  `batch_size`
2.  `gradient_checkpointing`
3.  `fp16` 


1.  `temperature` 0.7-0.9
2.  `max_new_tokens`
3. prompt


A:  `model_path` 

```bash
python scripts/Inference/run_inference.py --model-path ./path/to/your/model
```


A: 
-  (`batch_size > 1`)
-  (`fp16: true`)
-  `max_new_tokens`
- 


A:  `prompt/prompts.py`  `QUESTION_PROMPT`

```python
QUESTION_PROMPT = """prompt"""
```


A: 
-  `max_length`  `max_new_tokens`
- 
- 


1. ****:  `evaluate_on_dataset` 
2. ****: 
3. **Web**: Flask/FastAPIAPI


- `GRPOInference`: 
- `InferenceConfig`: 
- `run_inference.py`: 
- `inference_config.yaml`: 


