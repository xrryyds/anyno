import os
import json
import torch
import numpy as np
import logging
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed
)
from peft import PeftModel


from scripts import TakeExam, TeacherCorrecter
from utils import (
    FileIOUtils, 
    remove_null_hints, 
    filter_json_by_question_idx, 
    generate_irdcl_dataset,
    remove_null_hints
)
from configs import GRPOConfig
from data_math import Math_500, GSM8K


# =====================================================
# Logger Setup
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================
# Global Config
# =====================================================
exam_paper = FileIOUtils()
# model_path = "/mnt/petrelfs/wanhaiyuan/xrr/CELPO/model/OREAL/OREAL-7B"
# model_path = "/mnt/petrelfs/wanhaiyuan/xrr/CELPO/model/OREAL/OREAL-32B"
# model_path = "/root/autodl-tmp/Qwen2.5-Math-7B-Instruct/"
model_path = "/root/autodl-tmp/CELPO/model/OREAL/OREAL-7B"

def exam_roll_recheck_hints():
    try:
        logger.info("Step 1: Loading Dataset...")
        # 虽然这里读取了data_a，但在后续逻辑中主要使用 exam_paper.parse_hints_exam 解析出的数据
        with open(exam_paper.disadv_hints_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 解析原始输入数据，获取元数据（hints, from_entropy等）
        question_idx, _, question_with_hint, ref_solution, ref_answer, _, hints, from_entropy = exam_paper.parse_hints_exam(data)
        
        # 建立元数据映射字典，方便后续根据ID找回 hints 和 原始熵
        # Key: question_idx, Value: {hints, entropy}
        meta_map = {}
        for q_id, h, ent in zip(question_idx, hints, from_entropy):
            meta_map[q_id] = {'hints': h, 'orig_ent': ent}

        logger.info("Step 2: Student Rolling Exam...")
        student_exam = TakeExam(model_path=model_path)
        student_exam.exam_roll_k(question=question_with_hint, solution=ref_solution, answer=ref_answer, question_idx=question_idx)
        
        logger.info("Step 3: Teacher Grading...")
        teacher = TeacherCorrecter()
        # 获取批改结果
        _, correct_data = teacher.teacher_mark_paper(True)

        # Step 4: 处理 Correct Data (去重 + 保留最短答案)
        # 解包 correct_data: [ids, questions, student_answers, ref_solutions, ref_answers, entropies]
        c_ids, c_qs, c_ans, c_sols, c_refs, c_ents = correct_data
        
        best_candidates = {} # 用于去重的字典: {question_idx: data_item}

        for i in range(len(c_ids)):
            qid = c_ids[i]
            curr_ans = c_ans[i]
            
            # 构造要保存的数据项
            item = {
                "question_idx": qid,
                "question": c_qs[i],
                "hints": meta_map.get(qid, {}).get('hints', []),  # 找回对应的提示
                "student_answer": curr_ans,
                "ref_solution": c_sols[i],
                "ref_answer": c_refs[i],
                "entropy_original": meta_map.get(qid, {}).get('orig_ent', 0.0), # 找回原始熵
                "entropy_with_hints": c_ents[i],
                "success": True
            }
            
            # 去重逻辑：如果ID已存在，比较答案长度，保留更短的
            if qid not in best_candidates:
                best_candidates[qid] = item
            else:
                prev_len = len(best_candidates[qid]["student_answer"])
                curr_len = len(curr_ans)
                if curr_len < prev_len:
                    best_candidates[qid] = item # 更新为更短的答案

        # 将字典转回列表
        new_data_to_append = list(best_candidates.values())
        logger.info(f"Filtered {len(c_ids)} correct samples down to {len(new_data_to_append)} unique items (shortest answer strategy).")

        # Step 5: 追加写入文件
        target_path = exam_paper.adv_hints_dataset_path
        existing_data = []

        # 读取现有数据
        if os.path.exists(target_path):
            try:
                with open(target_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        logger.warning(f"Existing file {target_path} is not a list. Overwriting.")
                        existing_data = []
            except json.JSONDecodeError:
                logger.warning(f"Could not decode {target_path}. Starting with empty list.")
                existing_data = []

        # 合并数据
        final_data = existing_data + new_data_to_append

        exam_paper.save_results_to_json(final_data, exam_paper.adv_hints_dataset_path)
        
        logger.info(f"Successfully appended {len(new_data_to_append)} items to {target_path}. Total items: {len(final_data)}")
        
        return {
            'success': True,
            'processed_count': len(new_data_to_append)
        }

    except FileNotFoundError as e:
        error_msg = f'file not found: {e.filename}'
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    
    except json.JSONDecodeError as e:
        error_msg = f'JSON decode error: {str(e)}'
        logger.error(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    
    except Exception as e:
        error_msg = f'unknown error: {str(e)}'
        logger.error(error_msg)
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': error_msg
        }

def student_correct():
    logger.info("Step 1: Loading Dataset...")
    # 1. 加载原始带有提示的数据
    exam_paper.load_question_with_hints()
    # 解析数据：注意这里最后获取了 from_entropy (原始熵)
    question_idx, question, question_with_hint, ref_solution, ref_answer, _, hints, from_entropy = exam_paper.parse_hints_exam(exam_paper.question_with_hints)
   
    # 2. 学生考试 (使用带提示的题目进行推理)
    logger.info("Step 2: Student Taking Exam...")
    student_exam = TakeExam(model_path=model_path)
    # 这里的 exam 会计算并返回新的 entropy (虽然 exam 方法本身不返回，但结果会被保存并由 Teacher 读取)
    student_exam.exam(question=question_with_hint, solution=ref_solution, answer=ref_answer, question_idx=question_idx)

    # 3. 老师批改
    logger.info("Step 3: Teacher Grading...")
    teacher = TeacherCorrecter()
    incorrect_data, correct_data = teacher.teacher_mark_paper()
    
    # 解包批改结果
    # 结构: [ids, questions, student_answers, ref_solutions, ref_answers, entropies]
    # 这里我们要获取最后一位的 entropies (这是使用 Hints 后的熵)
    err_question_idx, _, err_answers, _, _, err_entropies = incorrect_data
    correct_question_idx, _, correct_answers, _, _, correct_entropies = correct_data

    # 4. 构建映射字典 (Question ID -> {Answer, Entropy})
    results_map = {}
    
    # 存入错题信息
    for q_id, s_ans, s_ent in zip(err_question_idx, err_answers, err_entropies):
        results_map[str(q_id)] = {
            "answer": s_ans,
            "entropy": s_ent
        }
        
    # 存入对题信息
    for q_id, s_ans, s_ent in zip(correct_question_idx, correct_answers, correct_entropies):
        results_map[str(q_id)] = {
            "answer": s_ans,
            "entropy": s_ent
        }

    # 创建错题 ID 集合，用于分类
    err_ids_set = set(str(x) for x in err_question_idx)
    # 已处理 ID 集合
    processed_ids_set = set(results_map.keys())

    # 5. 分类合并
    correct_group = []
    incorrect_group = []
    
    # 遍历原始数据，加入 from_entropy
    total_data = zip(question_idx, question, hints, ref_solution, ref_answer, from_entropy)
    
    for q_id, q, q_hints, r_sol, r_ans, orig_ent in total_data:
        str_qid = str(q_id)
        
        if str_qid not in processed_ids_set:
            logger.warning(f"Question ID {q_id} missing from exam results. Skipping.")
            continue
            
        # 获取考试结果
        exam_res = results_map[str_qid]
        s_ans = exam_res["answer"]
        hint_ent = exam_res["entropy"]

        item = {
            "question_idx": q_id,
            "question": q,
            "hints": q_hints,
            "student_answer": s_ans,
            "ref_solution": r_sol,
            "ref_answer": r_ans,
            "entropy_original": orig_ent,   # 原始熵
            "entropy_with_hints": hint_ent  # 使用 Hints 后的熵
        }

        # 分类逻辑
        if str_qid in err_ids_set:
            incorrect_group.append(item)
        else:
            correct_group.append(item)

    logger.info(f"Classification Done. Correct: {len(correct_group)}, Incorrect: {len(incorrect_group)}")

    # 6. 构造输出列表
    # 6.1 Teacher GRPO (包含 success 标记)
    data_for_teacher_grpo = []
    for item in correct_group:
        data_for_teacher_grpo.append({
            **item, # 包含 entropy 信息
            "success": True
        })
    for item in incorrect_group:
        data_for_teacher_grpo.append({
            **item,
            "success": False
        })
    
    # 6.2 Student Advantageous Hints (对题)
    data_for_student_adv_hints = correct_group # 结构已满足要求

    # 6.3 Student Disadvantageous Hints (错题)
    data_for_student_disadv_hints = incorrect_group # 结构已满足要求
    
    # 7. 保存
    adv_hints_dataset_path = exam_paper.adv_hints_dataset_path
    disadv_hints_dataset_path = exam_paper.disadv_hints_dataset_path
    grpo_dataset_path = exam_paper.grpo_dataset_path

    logger.info(f"Saving {len(data_for_teacher_grpo)} GRPO samples to {grpo_dataset_path}")
    logger.info(f"Saving {len(data_for_student_adv_hints)} Advantageous Hint samples to {adv_hints_dataset_path}")
    logger.info(f"Saving {len(data_for_student_disadv_hints)} Disadvantageous Hint samples to {disadv_hints_dataset_path}")

    exam_paper.save_results_to_json(data_for_teacher_grpo, grpo_dataset_path)
    exam_paper.save_results_to_json(data_for_student_adv_hints,  adv_hints_dataset_path)
    exam_paper.save_results_to_json(data_for_student_disadv_hints, disadv_hints_dataset_path)



def teacher_correct():
    teacher = TeacherCorrecter()
    teacher.teacher_mark_paper_with_save()
    teacher.teacher_hints()
    remove_null_hints(exam_paper.hints_file_path)
    filter_json_by_question_idx(exam_paper.exam_file_path, exam_paper.hints_file_path, exam_paper.corr_path)
    del teacher


def single_qusestion(qusetion):
    student_exam = TakeExam(model_path)
    return student_exam.answer_single_question(qusetion)


def student_first_take_exam_Math500():
    math_500 = Math_500()
    question = math_500.problems
    solution = math_500.solutions
    answer = math_500.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    take_exam = TakeExam(model_path=model_path)
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)




def student_first_take_exam_Gsm8k(train:bool = True):
    gsm8k = GSM8K(train=train)
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    take_exam = TakeExam(model_path)
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Gsm8k_test(use_lora:bool=False, lora_path:str=""):
    gsm8k = GSM8K(False)
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    take_exam = None
    if use_lora:
        take_exam = TakeExam(model_path=model_path,use_lora=True, adapter_path=lora_path)
    else:
        take_exam = TakeExam(model_path)

    take_exam.OUTPUT_JSON_PATH = take_exam.OUTPUT_JSON_PATH_TEST
    # take_exam.OUTPUT_JSON_PATH = take_exam.OUTPUT_JSON_PATH_EHC_TEST
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    
    take_exam.exam(question, solution, answer, question_idx)



def gen_IRDCL_dataset(batch_size):
    remove_null_hints(exam_paper.adv_hints_dataset_path)
    generate_irdcl_dataset(exam_paper.corr_path,
                        exam_paper.adv_hints_dataset_path,
                        exam_paper.disadv_hints_dataset_path,
                        exam_paper.irdcl_dataset_path,
                        batch_size,
                        0.5)


def exam_roll_recheck_mistake(use_lora:bool=False,lora_path:str=""):
    exam_paper.load_mistakes()
    m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution, m_entropy = exam_paper.parse_data(exam_paper.mistakes)
    
    logger.info(f"mistakes size: {len(m_question)}")

    take_exam = None
    if use_lora:
        take_exam = TakeExam(model_path=model_path,use_lora=True, adapter_path=lora_path)
    else:
        take_exam = TakeExam(model_path)
    take_exam.exam_roll_k(m_question, m_ref_solution, m_ref_answer, m_question_idx, 8, 0.7)

    teacher = TeacherCorrecter()
    
    _, correct_data = teacher.teacher_mark_paper(roll=True)
    correct_question_idx, _, _, _, _, _ = correct_data
    solved_ids = set(correct_question_idx)

    err_question_idx = []
    err_questions = []
    err_answers = []
    err_ref_answers = []
    err_ref_solutions = []
    err_entropy = []

    for i, idx in enumerate(m_question_idx):
        if idx not in solved_ids:
            err_question_idx.append(idx)
            err_questions.append(m_question[i])
            err_answers.append(m_answer[i])          
            err_ref_answers.append(m_ref_answer[i])
            err_ref_solutions.append(m_ref_solution[i])
            err_entropy.append(m_entropy[i])

    logger.info(f"Recheck Result -> Original: {len(m_question_idx)}, Solved: {len(solved_ids)}, Remaining: {len(err_question_idx)}")

    exam_paper.save_mistakes(
        err_question_idx, 
        err_questions, 
        err_answers, 
        err_ref_solutions, 
        err_ref_answers, 
        err_entropy
    )


def student_take_exam_Gsm8k_grpo_test():
    # 1. 准备数据
    logger.info("Loading Dataset...")
    gsm8k = GSM8K(False) # 确保这里的 GSM8K 类能正确引入
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    
    logger.info(f"Dataset Loaded: {len(question)} samples.")

    logger.info(f"Loading Base Model from {BASE_MODEL_PATH}...")
    take_exam = TakeExam(model_path=BASE_MODEL_PATH)

    # 3. 【核心步骤】加载 GRPO LoRA Adapter
    # 这步操作不会破坏 TakeExam 的其他功能，只是在运行时替换了内部的 model
    logger.info(f"Loading GRPO Adapter from {GRPO_ADAPTER_PATH}...")
    try:
        take_exam.model = PeftModel.from_pretrained(
            take_exam.model, 
            GRPO_ADAPTER_PATH,
            torch_dtype=torch.float16
        )
        take_exam.model.eval() # 切换到评估模式
        logger.info("Successfully loaded GRPO adapter!")
    except Exception as e:
        logger.error(f"Error loading adapter: {e}")
        logger.error("Ensure the path exists and contains adapter_config.json and adapter_model.safetensors")
        return

    # 4. 生成 Question ID (原有逻辑)
    question_idx = list(range(len(question)))

    # 5. 运行测试
    logger.info("Starting Inference...")
    accuracy = take_exam.exam_test(question, solution, answer, question_idx)
    
    logger.info(f"Final GRPO Model Accuracy: {accuracy:.2%}")


def take_exam_gsm8k_after_EHC(lora_path:str):
    exam_paper_easl = TakeExam(model_path=model_path, use_lora=True, adapter_path=lora_path)
    gsm8k = GSM8K()
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    exam_paper_easl.exam(question, solution, answer, question_idx)

def take_exam_MATH500_after_EHC(lora_path:str):
    exam_paper_easl = TakeExam(model_path=model_path, use_lora=True, adapter_path=lora_path)
    math_500 = Math_500()
    question = math_500.problems
    solution = math_500.solutions
    answer = math_500.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    exam_paper_easl.exam(question, solution, answer, question_idx)

if __name__ == "__main__":
    # #1. student first take exam
    # student_first_take_exam_Math500()
    # student_first_take_exam_Gsm8k(False)


    # #2. teacher judges
    teacher = TeacherCorrecter()
    # teacher.teacher_mark_paper_with_save()
    # teacher.check_answers_equivalence()

    # 3. student roll on mistake
    exam_roll_recheck_mistake()

    # 4. teacher_give_hints
    # teacher.teacher_hints() 

    # student_correct()
    # exam_roll_recheck_hints()
    # 3. gen dataset
    # gen_IRDCL_dataset(8)
# python -m scripts.train.student_train
    # 4. check
    # take_exam_MATH500_after_EHC("/root/autodl-tmp/CELPO/output/hint_sft_0203_0351")
    # teacher.teacher_mark_paper_with_save()
    # teacher.check_answers_equivalence()
    # exam_roll_recheck_mistake(True,"/root/autodl-tmp/CELPO/output/hint_sft_0203_0351")

    # student_take_exam_Gsm8k_test(True, "/root/autodl-tmp/CELPO/output/hint_sft_0203_0351")
    # teacher.teacher_mark_paper_with_save()
    # teacher.check_answers_equivalence()
    # exam_roll_recheck_mistake(True,"/root/autodl-tmp/CELPO/output/hint_sft_0203_0351")
    #####################################################################################################
    
    # BASE_MODEL_PATH = "/root/autodl-tmp/CELPO/model/Qwen/Qwen2.5-Math-7B-Instruct"
    # # GRPO 训练保存的 checkpoint 路径，通常是 epoch_X
    # GRPO_ADAPTER_PATH = "/root/autodl-tmp/CELPO/output/hint_grpo/epoch_2" 
    
    # if not os.path.exists(GRPO_ADAPTER_PATH):
    #     logger.warning(f"Adapter path {GRPO_ADAPTER_PATH} does not exist. Check if training finished.")
    # else:
    #     # student_take_exam_Gsm8k_grpo_test()
    #     pass

