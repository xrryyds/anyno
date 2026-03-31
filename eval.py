import os
import sys
import re
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from data_math import Math_500
# ==========================================
# 0. 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 全局推理序列长度超参数（评测用）
MAX_SEQ_LENGTH = 2048

def extract_answer(text):
    if not text: return ""
    
    # 策略 1: LaTeX boxed
    # 解决嵌套括号问题比较麻烦，这里用简化的正则
    matches = re.findall(r"\\boxed\{(.+?)\}", text)
    if matches:
        # 取最后一个 boxed
        return matches[-1].strip()
    
    # 策略 2: Answer: X
    parts = text.split("Answer:")
    if len(parts) > 1:
        return parts[-1].strip().split('\n')[0]
        
    return ""

def check_correctness(pred_str, gt_str):
    # 1. 预处理：去空格，统一格式
    def normalize(s):
        s = str(s).strip()
        # 移除 latex 格式符号
        s = s.replace('$', '').replace('\\', '')
        return s
    
    pred = normalize(pred_str)
    gt = normalize(gt_str)
    
    if pred == gt:
        return True
    
    # 简单的包含匹配 (非常宽松)
    if gt in pred:
        return True
        
    return False

# ==========================================
# 3. 封装评估逻辑 (类结构)
# ==========================================

class StudentEvaluator:
    def __init__(self, base_model_path, adapter_path, gpu_memory_utilization=0.9):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        
        logger.info(f"Initializing vLLM with Base: {base_model_path} | Adapter: {adapter_path}")
        
        # 初始化 vLLM
        self.llm = LLM(
            model=base_model_path,
            enable_lora=True,
            max_lora_rank=64,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=True
        )
        
        # 使用全局推理长度超参数
        self.sampling_params = SamplingParams(
            temperature=0.0, # 评测通常用 Greedy Search (temp=0)
            max_tokens=MAX_SEQ_LENGTH,
        )
        
        self.adapter_name = "grpo_adapter"

    def batch_generate(self, questions, batch_size=32):
        # 构造 Prompt
        prompts = questions
        
        # 生成
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            lora_request=LoRARequest(self.adapter_name, 1, self.adapter_path),
            use_tqdm=True
        )
        
        generated_texts = [output.outputs[0].text for output in outputs]
        return generated_texts

    def evaluate(self, questions, solutions, answers):
        logger.info("Starting Evaluation...")
        
        generated_texts = self.batch_generate(questions)
        
        correct_count = 0
        total = len(questions)
        results = []
        
        logger.info("Checking answers...")
        for i in range(total):
            gen_text = generated_texts[i]
            gt_ans = answers[i]
            
            # 提取预测答案
            pred_ans = extract_answer(gen_text)
            
            # 判题
            is_correct = check_correctness(pred_ans, gt_ans)
            if is_correct:
                correct_count += 1
                
            results.append({
                "question": questions[i],
                "generated": gen_text,
                "pred_extract": pred_ans,
                "ground_truth": gt_ans,
                "is_correct": is_correct
            })
            
            if i < 3: 
                logger.info(f"\n[Sample {i}]")
                logger.info(f"Gen: {gen_text[:100]}...")
                logger.info(f"Pred: {pred_ans} | GT: {gt_ans} | Correct: {is_correct}")

        accuracy = correct_count / total * 100
        logger.info("=" * 40)
        logger.info(f"Math-500 Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
        logger.info("=" * 40)
        
        return accuracy, results

# ==========================================
# 4. 主函数 (模仿你的调用方式)
# ==========================================

def student_first_take_exam_Math500():
    # 1. 配置路径
    base_model_path = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"
    # 请确认这是你的 adapter 真实路径
    grpo_adapter_path = "/root/autodl-tmp/CELPO/output/grpo_vllm_output/checkpoint-epoch-1" 

    # 2. 加载数据集
    try:
        math_500 = Math_500()
        question = math_500.problems
        solution = math_500.solutions
        answer = math_500.answers
        
        logger.info(f"Dataset Size: {len(question)}")
    except Exception as e:
        logger.error("Failed to load Math500 dataset. Ensure you have 'datasets' installed or implement local loading.")
        return

    # 3. 初始化评测器
    evaluator = StudentEvaluator(
        base_model_path=base_model_path, 
        adapter_path=grpo_adapter_path
    )
    
    accuracy, results = evaluator.evaluate(question, solution, answer)
    
    print(accuracy)
    # 5. 保存结果 (可选)
    import json
    with open("math500_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to math500_results.json")

if __name__ == "__main__":
    student_first_take_exam_Math500()
