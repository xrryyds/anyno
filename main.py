import os
import json
import torch
import numpy as np
import logging
from tqdm import tqdm
from scripts import run_sira_training, run_sft_training, run_grpo_training
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
    remove_null_hints,
)
from data_math import Math_500, GSM8K, AIME2024, Math_All, Math_Subset


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
# model_path = "/mnt/petrelfs/wanhaiyuan/xrr/CELPO/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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


def student_take_exam_Math500():
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



def student_take_exam_Math_sub(train:bool = True, subset:str="all", lora_path:str = None):
    data = Math_All(subset_name=subset,train=train)
    question = data.problems
    solution = data.solutions
    answer = data.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    take_exam = None
    if lora_path:
        take_exam = TakeExam(model_path, use_lora=True, adapter_path = lora_path)
    else:
        take_exam = TakeExam(model_path)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Gsm8k(train:bool = True, lora_path:str = None):
    gsm8k = GSM8K(train=train)
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers
    
    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")
    
    take_exam = None
    if lora_path:
        take_exam = TakeExam(model_path, use_lora=True, adapter_path = lora_path)
    else:
        take_exam = TakeExam(model_path)

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
                        0.5, 1)


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


def sft_on_adv_Data():
    try:
        with open(exam_paper.adv_hints_dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"load fail: {e}")
    _, question,  _,  _, ref_solution,  _ = exam_paper.parse_data(data)
    run_sft_training(model_url=model_path, question_list=question, answer_list=ref_solution)


def count_common_questions(file_corr = exam_paper.corr_path, file_hints = exam_paper.hints_file_path):
    try:
        # 读取 corr.json 文件
        with open(file_corr, 'r', encoding='utf-8') as f:
            corr_data = json.load(f)
            
        # 读取 adv_hints.json 文件
        with open(file_hints, 'r', encoding='utf-8') as f:
            hints_data = json.load(f)
            
        # 提取 question_idx 集合
        # 假设文件结构是列表，每个元素是包含 'question_idx' 的字典
        corr_ids = {item['question_idx'] for item in corr_data if 'question_idx' in item}
        hints_ids = {item['question_idx'] for item in hints_data if 'question_idx' in item}
        
        # 计算交集
        common_ids = corr_ids.intersection(hints_ids)
        
        # 返回相同 question_idx 的数量
        print(len(common_ids))

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return 0
    except json.JSONDecodeError:
        print("错误: 文件不是有效的 JSON 格式")
        return 0
    except Exception as e:
        print(f"发生未知错误: {e}")
        return 0

def sft_on_mistakes(model_path: str):
    
    # 2. 加载错题
    logger.info("Loading mistakes...")
    if not exam_paper.load_mistakes():
        logger.error("Failed to load mistakes. Aborting SFT.")
        return

    _, questions, _, _, ref_solutions, _ = exam_paper.parse_data(exam_paper.mistakes)

    if not questions or len(questions) == 0:
        logger.warning("No questions found in mistake file.")
        return

    valid_questions = []
    valid_solutions = []

    for q, sol in zip(questions, ref_solutions):
        if q and sol:
            valid_questions.append(q)
            valid_solutions.append(sol)
    
    logger.info(f"Prepared {len(valid_questions)} pairs for training (Model will learn 'ref_solution').")

    run_sft_training(
        model_url=model_path,
        question_list=valid_questions,
        answer_list=valid_solutions, # 使用标准答案进行训练
        num_train_epochs=1           # 针对少量错题，通常跑 3-5 个 epoch
    )

def grpo_on_MATH(lora_path:str, subset:str ="all"):
    data = Math_All(subset_name=subset,train=True)
    question = data.problems
    answer = data.answers
    run_grpo_training(model_path=model_path, sft_lora_path=lora_path, questions=question, answers=answer)

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py
    # CUDA_VISIBLE_DEVICES=0  python main.py
    # #1. student first take exam
    # student_take_exam_Math500()
    # student_take_exam_Gsm8k(False)
    # student_take_exam_Math_sub(train=True)

    # #2. teacher judges
    teacher = TeacherCorrecter()
    # teacher.teacher_mark_paper_with_save()

    # 3. student roll on mistake
    # exam_roll_recheck_mistake()
    # teacher.check_answers_equivalence()

    # 4. teacher_give_hints
    # teacher.teacher_hints() 

    # student_correct()
    # exam_roll_recheck_hints()

    # ** sft
    # sft_on_adv_Data()
    
    # 3. gen dataset
    # gen_IRDCL_dataset(16)
    # run_sira_training(model_path=model_path)
    # 4. check
    student_take_exam_Math_sub(train=False, lora_path="/root/autodl-tmp/CELPO/output/sft_lora_checkpoints/final_adapter")
    # student_take_exam_Gsm8k(train=False, lora_path="/mnt/petrelfs/wanhaiyuan/xrr/CELPO/output/sira_sft_0204_2128")
    teacher.teacher_mark_paper_with_save()
    # count_common_questions()
    # teacher.check_answers_equivalence()
    # exam_roll_recheck_mistake(True, "/root/autodl-tmp/CELPO/output/sft_lora_checkpoints/final_adapter")
    
    # grpo_on_MATH("/root/autodl-tmp/CELPO/output/sira_sft_0206_1232")

    #####################################################################################################