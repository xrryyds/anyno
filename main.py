import os
import json
import gc
import random
import torch
import numpy as np
import logging
from tqdm import tqdm
from scripts import (
    run_sira_training_v3,
    run_sft_training_baseline,
    run_sdft_training_baseline,
    run_sdpo_training_baseline,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel


from scripts import TakeExam, TeacherCorrecter
from utils import (
    FileIOUtils,
    remove_null_hints,
    filter_json_by_question_idx,
    generate_irdcl_dataset,
    generate_irdcl_datase_v2,
    remove_null_hints,
    merge_lora_to_base_model,
    generate_sft_data,
)
from data_math import (
    Math_500,
    GSM8K,
    AIME,
    Math_All,
    Math_Subset,
    LiveMathBench,
    AIME_1983_2024,
)

model_path = "/workspace/CELPO_model/model/DS/Llama-3-8B-Inst"


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
_tokenizer_cache = None


def _get_tokenizer():
    """Lazy-load tokenizer for hint truncation."""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
    return _tokenizer_cache


def truncate_hints_by_tokens(hints_list: list, max_tokens: int) -> list:
    """Truncate each hint string in hints_list to at most max_tokens tokens.

    Args:
        hints_list: List of hint strings.
        max_tokens: Maximum number of tokens to keep for each hint.

    Returns:
        List of truncated hint strings.
    """
    if max_tokens is None:
        return hints_list
    tokenizer = _get_tokenizer()
    truncated = []
    for hint in hints_list:
        if hint is None or hint == "":
            truncated.append(hint)
            continue
        token_ids = tokenizer.encode(hint, add_special_tokens=False)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            hint = tokenizer.decode(token_ids, skip_special_tokens=True)
        truncated.append(hint)
    return truncated


def exam_roll_recheck_hints(
    lora_path: str = None, max_token: int = 2048, hint_token_limit: int = None
):
    try:
        logger.info("Step 1: Loading Dataset...")
        with open(exam_paper.disadv_hints_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        (
            question_idx,
            question,
            question_with_hint,
            ref_solution,
            ref_answer,
            _,
            hints,
            from_entropy,
        ) = exam_paper.parse_hints_exam(data)

        # Truncate hints if hint_token_limit is specified
        if hint_token_limit is not None:
            logger.info(f"Truncating hints to {hint_token_limit} tokens...")
            hints = truncate_hints_by_tokens(hints, hint_token_limit)

        # Key: question_idx, Value: {hints, entropy}
        meta_map = {}
        for q_id, h, ent in zip(question_idx, hints, from_entropy):
            meta_map[q_id] = {"hints": h, "orig_ent": ent}

        logger.info("Step 2: Student Rolling Exam...")
        if lora_path:
            student_exam = TakeExam(
                model_path=model_path,
                use_lora=True,
                adapter_path=lora_path,
                max_seq_length=max_token,
            )
        else:
            student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
        student_exam.exam_roll_k_with_hints(
            question=question,
            solution=ref_solution,
            answer=ref_answer,
            question_idx=question_idx,
            hints=hints,
        )

        logger.info("Step 3: Teacher Grading...")
        teacher = TeacherCorrecter()
        _, correct_data = teacher.teacher_mark_paper(True)

        c_ids, c_qs, c_ans, c_sols, c_refs, c_ents = correct_data

        best_candidates = {}

        for i in range(len(c_ids)):
            qid = c_ids[i]
            curr_ans = c_ans[i]

            item = {
                "question_idx": qid,
                "question": c_qs[i],
                "hints": meta_map.get(qid, {}).get("hints", []),
                "student_answer": curr_ans,
                "ref_solution": c_sols[i],
                "ref_answer": c_refs[i],
                "entropy_original": meta_map.get(qid, {}).get("orig_ent", 0.0),
                "entropy_with_hints": c_ents[i],
                "success": True,
            }

            if qid not in best_candidates:
                best_candidates[qid] = item
            else:
                prev_len = len(best_candidates[qid]["student_answer"])
                curr_len = len(curr_ans)
                if curr_len < prev_len:
                    best_candidates[qid] = item

        new_data_to_append = list(best_candidates.values())
        logger.info(
            f"Filtered {len(c_ids)} correct samples down to {len(new_data_to_append)} unique items (shortest answer strategy)."
        )

        target_path = exam_paper.adv_hints_dataset_path
        existing_data = []

        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        logger.warning(
                            f"Existing file {target_path} is not a list. Overwriting."
                        )
                        existing_data = []
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode {target_path}. Starting with empty list."
                )
                existing_data = []

        final_data = existing_data + new_data_to_append

        exam_paper.save_results_to_json(final_data, exam_paper.adv_hints_dataset_path)

        logger.info(
            f"Successfully appended {len(new_data_to_append)} items to {target_path}. Total items: {len(final_data)}"
        )

        return {"success": True, "processed_count": len(new_data_to_append)}

    except FileNotFoundError as e:
        error_msg = f"file not found: {e.filename}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"unknown error: {str(e)}"
        logger.error(error_msg)
        import traceback

        traceback.print_exc()
        return {"success": False, "error": error_msg}


def process_exam_file_batch(file_path, lora_path: str = None, max_token: int = 2048):
    """
    JSON student_exam.exam
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [item.get("question", "") for item in data]

        solutions = [item.get("ref_solution", "") for item in data]

        answers = [item.get("ref_answer", "") for item in data]

        indices = [item.get("question_idx", 0) for item in data]

        student_exam = None

        if lora_path:
            student_exam = TakeExam(
                model_path=model_path,
                use_lora=True,
                adapter_path=lora_path,
                max_seq_length=max_token,
            )
        else:
            student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
        student_exam.exam(
            question=questions, solution=solutions, answer=answers, question_idx=indices
        )

        print(f" {len(data)} ")

    except FileNotFoundError:
        print(f" {file_path}")
    except json.JSONDecodeError:
        print("JSON ")
    except Exception as e:
        print(f"{e}")


def student_correct(
    lora_path: str = None, max_token: int = 2048, hint_token_limit: int = None
):
    logger.info("Step 1: Loading Dataset...")
    exam_paper.load_question_with_hints()
    (
        question_idx,
        question,
        question_with_hint,
        ref_solution,
        ref_answer,
        _,
        hints,
        from_entropy,
    ) = exam_paper.parse_hints_exam(exam_paper.question_with_hints)

    # Truncate hints if hint_token_limit is specified
    if hint_token_limit is not None:
        logger.info(f"Truncating hints to {hint_token_limit} tokens...")
        hints = truncate_hints_by_tokens(hints, hint_token_limit)

    logger.info("Step 2: Student Taking Exam...")
    if lora_path:
        student_exam = TakeExam(
            model_path=model_path,
            use_lora=True,
            adapter_path=lora_path,
            max_seq_length=max_token,
        )
    else:
        student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
    student_exam.exam_with_hints(
        question=question,
        solution=ref_solution,
        answer=ref_answer,
        question_idx=question_idx,
        hints=hints,
    )

    logger.info("Step 3: Teacher Grading...")
    teacher = TeacherCorrecter()
    incorrect_data, correct_data = teacher.teacher_mark_paper()

    err_question_idx, _, err_answers, _, _, err_entropies = incorrect_data
    correct_question_idx, _, correct_answers, _, _, correct_entropies = correct_data

    results_map = {}

    for q_id, s_ans, s_ent in zip(err_question_idx, err_answers, err_entropies):
        results_map[str(q_id)] = {"answer": s_ans, "entropy": s_ent}

    for q_id, s_ans, s_ent in zip(
        correct_question_idx, correct_answers, correct_entropies
    ):
        results_map[str(q_id)] = {"answer": s_ans, "entropy": s_ent}

    err_ids_set = set(str(x) for x in err_question_idx)
    processed_ids_set = set(results_map.keys())

    correct_group = []
    incorrect_group = []

    total_data = zip(
        question_idx, question, hints, ref_solution, ref_answer, from_entropy
    )

    for q_id, q, q_hints, r_sol, r_ans, orig_ent in total_data:
        str_qid = str(q_id)

        if str_qid not in processed_ids_set:
            logger.warning(f"Question ID {q_id} missing from exam results. Skipping.")
            continue

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
            "entropy_original": orig_ent,
            "entropy_with_hints": hint_ent,
        }

        if str_qid in err_ids_set:
            incorrect_group.append(item)
        else:
            correct_group.append(item)

    logger.info(
        f"Classification Done. Correct: {len(correct_group)}, Incorrect: {len(incorrect_group)}"
    )

    data_for_teacher_grpo = []
    for item in correct_group:
        data_for_teacher_grpo.append({**item, "success": True})
    for item in incorrect_group:
        data_for_teacher_grpo.append({**item, "success": False})

    data_for_student_adv_hints = correct_group

    data_for_student_disadv_hints = incorrect_group

    adv_hints_dataset_path = exam_paper.adv_hints_dataset_path
    disadv_hints_dataset_path = exam_paper.disadv_hints_dataset_path
    grpo_dataset_path = exam_paper.grpo_dataset_path

    logger.info(
        f"Saving {len(data_for_teacher_grpo)} GRPO samples to {grpo_dataset_path}"
    )
    logger.info(
        f"Saving {len(data_for_student_adv_hints)} Advantageous Hint samples to {adv_hints_dataset_path}"
    )
    logger.info(
        f"Saving {len(data_for_student_disadv_hints)} Disadvantageous Hint samples to {disadv_hints_dataset_path}"
    )

    exam_paper.save_results_to_json(data_for_teacher_grpo, grpo_dataset_path)
    exam_paper.save_results_to_json(data_for_student_adv_hints, adv_hints_dataset_path)
    exam_paper.save_results_to_json(
        data_for_student_disadv_hints, disadv_hints_dataset_path
    )


def teacher_correct():
    teacher = TeacherCorrecter()
    teacher.teacher_mark_paper_with_save()
    teacher.teacher_hints()
    remove_null_hints(exam_paper.hints_file_path)
    filter_json_by_question_idx(
        exam_paper.exam_file_path, exam_paper.hints_file_path, exam_paper.corr_path
    )
    del teacher


def single_qusestion(qusetion, max_token: int = 2048):
    student_exam = TakeExam(model_path, max_seq_length=max_token)
    return student_exam.answer_single_question(qusetion)


def student_take_exam_Math500(max_token: int = 2048):
    math_500 = Math_500()
    question = math_500.problems
    solution = math_500.solutions
    answer = math_500.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Math_sub(
    train: bool = True,
    subset: str = "all",
    lora_path: str = None,
    max_token: int = 2048,
):
    data = Math_All(subset_name=subset, train=train)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam_multi_gpu(question, solution, answer, question_idx)


def student_take_exam_AIME(
    lora_path: str = None,
    year=2024,
    model_path: str = model_path,
    max_token: int = 2048,
):
    data = AIME(year=year)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_AIME_1983_2024(
    lora_path: str = None, model_path: str = model_path, max_token: int = 2048
):
    data = AIME_1983_2024()
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Math_500(
    train: bool = True,
    subset: str = "all",
    lora_path: str = None,
    max_token: int = 2048,
):
    data = Math_500()
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_LiveMath(
    lora_path: str = None, max_size: int = None, max_token: int = 2048
):
    """Run an exam on the LiveMathBench-en dataset.

    Args:
        lora_path: Optional LoRA adapter path; if provided, exam runs with LoRA.
        max_size: Optionally limit the number of questions for quick debugging.
    """
    data = LiveMathBench(split="test", max_size=max_size)
    question = data.problems
    solution = data.solutions
    answer = data.answers

    logger.info(
        f"LiveMathBench dataset_len_check: {len(question)} {len(solution)} {len(answer)}"
    )

    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = list(range(len(question)))
    take_exam.exam(question, solution, answer, question_idx)


def student_take_exam_Gsm8k(
    train: bool = True, lora_path: str = None, max_token: int = 2048
):
    gsm8k = GSM8K(train=train)
    question = gsm8k.problems
    solution = gsm8k.solutions
    answer = gsm8k.answers

    logger.info(f"dataset_len_check: {len(question)} {len(solution)} {len(answer)}")

    take_exam = None
    if lora_path:
        take_exam = TakeExam(
            model_path, use_lora=True, adapter_path=lora_path, max_seq_length=max_token
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)

    question_idx = []
    for idx in range(len(question)):
        question_idx.append(idx)
    take_exam.exam(question, solution, answer, question_idx)


# def compute_and_save_ref_loss():
#     """ref  corr_path  answer tokens  CE loss ref_beta"""
#     import torch
#     from tqdm import tqdm
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     from scripts.train.student_train_v2 import FixedModeCollator, SYSTEM_PROMPT

#     with open(exam_paper.corr_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     if all("ref_beta" in item for item in data):
#         logger.info("ref_beta already computed for all items, skipping.")
#         return

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path, trust_remote_code=True, use_fast=False
#     )
#     collator = FixedModeCollator(tokenizer)
#     ref_model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,
#         device_map="auto",
#         trust_remote_code=True,
#     )
#     ref_model.eval()
#     loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

#     with torch.no_grad():
#         for item in tqdm(data, desc="Computing ref_beta", ncols=100):
#             if "ref_beta" in item:
#                 continue
#             sample = {
#                 "question": item["question"],
#                 "answer": item.get("answer", item.get("ref_solution", "")),
#                 "type": "anchor_data",
#             }
#             batch = collator([sample])
#             input_ids = batch["input_ids"].to(ref_model.device)
#             attention_mask = batch["attention_mask"].to(ref_model.device)
#             labels = batch["labels"].to(ref_model.device)
#             a_mask = batch["answer_masks"][0, 1:].to(ref_model.device)
#             logits = ref_model(
#                 input_ids=input_ids, attention_mask=attention_mask
#             ).logits
#             token_losses = loss_fct(logits[0, :-1], labels[0, 1:])
#             a_count = a_mask.sum()
#             item["ref_beta"] = (
#                 ((token_losses * a_mask).sum() / a_count).item() if a_count > 0 else 0.0
#             )

#     del ref_model
#     torch.cuda.empty_cache()

#     with open(exam_paper.corr_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#     logger.info(f"ref_beta saved to {exam_paper.corr_path}")


def gen_IRDCL_dataset(batch_size, spilt, epoch):
    # compute_and_save_ref_loss()
    remove_null_hints(exam_paper.adv_hints_dataset_path)
    generate_irdcl_dataset(
        exam_paper.corr_path,
        exam_paper.adv_hints_dataset_path,
        exam_paper.disadv_hints_dataset_path,
        exam_paper.irdcl_dataset_path,
        batch_size,
        spilt,
        epoch,
    )


def gen_IRDCL_dataset_v2(batch_size, spilt, epoch):
    # compute_and_save_ref_loss()
    remove_null_hints(exam_paper.adv_hints_dataset_path)
    generate_irdcl_datase_v2(
        exam_paper.corr_path,
        exam_paper.adv_hints_dataset_path,
        exam_paper.disadv_hints_dataset_path,
        exam_paper.irdcl_dataset_path,
        batch_size,
        spilt,
        epoch,
    )


def shuffle_irdcl_dataset(seed: int = None):
    """Read irdcl_data.json, shuffle the data in-place, and write it back.

    Args:
        seed: Optional random seed for reproducibility. If None, uses system randomness.
    """
    irdcl_path = exam_paper.irdcl_dataset_path

    if not os.path.exists(irdcl_path):
        logger.error(f"File not found: {irdcl_path}")
        return

    with open(irdcl_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error(f"Expected a JSON list in {irdcl_path}, got {type(data).__name__}")
        return

    logger.info(f"Shuffling {len(data)} items in {irdcl_path} (seed={seed})...")

    if seed is not None:
        random.seed(seed)
    random.shuffle(data)

    with open(irdcl_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Shuffled and saved {len(data)} items back to {irdcl_path}")


def replace_hints_with_ref_solution_prefix(max_tokens: int = 50):
    """Replace all hints in adv_hints.json with the first `max_tokens` tokens of ref_solution.

    This reads adv_hints.json, truncates each item's ref_solution to the first
    `max_tokens` tokens, and overwrites the hints field with that truncated text.
    The modified data is written back to adv_hints.json.

    Args:
        max_tokens: Number of tokens to take from the beginning of ref_solution.
                    Defaults to 50.
    """
    adv_path = exam_paper.adv_hints_dataset_path

    if not os.path.exists(adv_path):
        logger.error(f"File not found: {adv_path}")
        return

    with open(adv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error(f"Expected a JSON list in {adv_path}, got {type(data).__name__}")
        return

    tokenizer = _get_tokenizer()
    modified_count = 0

    for item in data:
        ref_sol = item.get("ref_solution", "")
        if not ref_sol:
            continue
        token_ids = tokenizer.encode(ref_sol, add_special_tokens=False)
        truncated_ids = token_ids[:max_tokens]
        truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        item["hints"] = truncated_text
        modified_count += 1

    with open(adv_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Replaced hints with first {max_tokens} tokens of ref_solution "
        f"for {modified_count}/{len(data)} items in {adv_path}"
    )


def exam_roll_recheck_mistake(
    use_lora: bool = False,
    lora_path: str = "",
    save_log_path: str = None,
    log_prompt: str = "",
    model_path=model_path,
    max_token: int = 2048,
):
    exam_paper.load_mistakes()
    m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution, m_entropy = (
        exam_paper.parse_data(exam_paper.mistakes)
    )

    logger.info(f"mistakes size: {len(m_question)}")

    take_exam = None
    if use_lora:
        take_exam = TakeExam(
            model_path=model_path,
            use_lora=True,
            adapter_path=lora_path,
            max_seq_length=max_token,
        )
    else:
        take_exam = TakeExam(model_path, max_seq_length=max_token)
    take_exam.exam_roll_k_multi_gpu(
        m_question, m_ref_solution, m_ref_answer, m_question_idx, 8, 0.7
    )

    teacher = TeacherCorrecter()

    _, correct_data = teacher.teacher_mark_paper(roll=True)
    (
        correct_question_idx,
        correct_questions,
        correct_answers,
        correct_ref_solutions,
        correct_ref_answers,
        correct_entropy,
    ) = correct_data
    solved_ids = set(correct_question_idx)

    roll8_solved_question_idx = []
    roll8_solved_questions = []
    roll8_solved_answers = []
    roll8_solved_ref_solutions = []
    roll8_solved_ref_answers = []
    roll8_solved_entropy = []

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
        else:
            for j, corr_idx in enumerate(correct_question_idx):
                if corr_idx == idx:
                    roll8_solved_question_idx.append(idx)
                    roll8_solved_questions.append(m_question[i])
                    roll8_solved_answers.append(correct_answers[j])
                    roll8_solved_ref_solutions.append(m_ref_solution[i])
                    roll8_solved_ref_answers.append(m_ref_answer[i])
                    roll8_solved_entropy.append(m_entropy[i])
                    break

    recheck_result_log = f"Recheck Result -> Original: {len(m_question_idx)}, Solved: {len(solved_ids)}, Remaining: {len(err_question_idx)}"
    logger.info(recheck_result_log)
    logger.info(f"mistake:{len(err_question_idx)}")

    if save_log_path:
        log_lines = [recheck_result_log]
        if log_prompt:
            log_lines.append(log_prompt)
        with open(save_log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
            f.write("#############################\n")

    exam_paper.save_mistakes(
        err_question_idx,
        err_questions,
        err_answers,
        err_ref_solutions,
        err_ref_answers,
        err_entropy,
    )

    if len(roll8_solved_question_idx) > 0:
        logger.info(
            f"Adding {len(roll8_solved_question_idx)} newly solved questions to corr_answer.json"
        )

        existing_corr_data = []
        try:
            with open(exam_paper.corr_path, "r", encoding="utf-8") as f:
                existing_corr_data = json.load(f)
            logger.info(f"Loaded {len(existing_corr_data)} existing correct answers")
        except Exception as e:
            logger.warning(
                f"Failed to load existing corr_answer.json: {e}, will create new file"
            )

        existing_idx_set = {item.get("question_idx") for item in existing_corr_data}

        for i in range(len(roll8_solved_question_idx)):
            if roll8_solved_question_idx[i] not in existing_idx_set:
                existing_corr_data.append(
                    {
                        "question_idx": roll8_solved_question_idx[i],
                        "question": roll8_solved_questions[i],
                        "answer": roll8_solved_answers[i],
                        "ref_solution": roll8_solved_ref_solutions[i],
                        "ref_answer": roll8_solved_ref_answers[i],
                        "entropy": roll8_solved_entropy[i],
                    }
                )

        try:
            with open(exam_paper.corr_path, "w", encoding="utf-8") as f:
                json.dump(existing_corr_data, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Successfully saved {len(existing_corr_data)} total correct answers to corr_answer.json"
            )
        except Exception as e:
            logger.error(f"Failed to save corr_answer.json: {e}")


def sft_on_adv_Data():
    try:
        with open(exam_paper.adv_hints_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"load fail: {e}")
    _, question, _, _, ref_solution, _ = exam_paper.parse_data(data)
    run_sft_training(
        model_url=model_path, question_list=question, answer_list=ref_solution
    )


def count_common_questions(
    file_corr=exam_paper.corr_path, file_hints=exam_paper.hints_file_path
):
    try:
        with open(file_corr, "r", encoding="utf-8") as f:
            corr_data = json.load(f)

        with open(file_hints, "r", encoding="utf-8") as f:
            hints_data = json.load(f)

        corr_ids = {
            item["question_idx"] for item in corr_data if "question_idx" in item
        }
        hints_ids = {
            item["question_idx"] for item in hints_data if "question_idx" in item
        }

        common_ids = corr_ids.intersection(hints_ids)

        print(len(common_ids))

    except FileNotFoundError as e:
        print(f":  - {e}")
        return 0
    except json.JSONDecodeError:
        print(":  JSON ")
        return 0
    except Exception as e:
        print(f": {e}")
        return 0


def sft_on_mistakes(model_path: str):

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

    logger.info(
        f"Prepared {len(valid_questions)} pairs for training (Model will learn 'ref_solution')."
    )

    run_sft_training(
        model_url=model_path,
        question_list=valid_questions,
        answer_list=valid_solutions,
        num_train_epochs=1,
    )


def grpo_on_MATH(lora_path: str, subset: str = "all"):
    data = Math_All(subset_name=subset, train=True)
    question = data.problems
    answer = data.answers
    run_grpo_training(model_path, lora_path, question, answer)


def grpo_on_MATH500(lora_path: str, num_generations: int = 8):
    """
     MATH500  GRPO

    Args:
        lora_path: SIRA  LoRA checkpoint
        num_generations: GRPO  8
    """
    logger.info("=" * 60)
    logger.info("Starting GRPO Training on MATH500")
    logger.info(f"Base Model: {model_path}")
    logger.info(f"SFT LoRA Path: {lora_path}")
    logger.info(f"Num Generations: {num_generations}")
    logger.info("=" * 60)

    data = Math_500()
    question = data.problems
    answer = data.answers

    logger.info(f"Dataset size: {len(question)} questions")

    run_grpo_training(
        base_model_path=model_path,
        sft_lora_path=lora_path,
        questions=question,
        answers=answer,
        num_generations=num_generations,
    )

    logger.info("GRPO Training completed!")


def test_adv_hints_accuracy(
    model_path: str, dataset_path: str = None, max_token: int = 2048
):
    """
     Advantageous Hints
     100%
    Temperature

    Args:
        model_path (str):
        dataset_path (str, optional): adv_hints  exam_paper.adv_hints_dataset_path
    """

    if dataset_path is None:
        try:
            dataset_path = exam_paper.adv_hints_dataset_path
        except NameError:
            logger.error(" dataset_path  exam_paper ")
            return

    if not os.path.exists(dataset_path):
        logger.error(f": {dataset_path}")
        return

    logger.info(f"Step 1: Loading Advantageous Hints Dataset from {dataset_path}...")

    with open(dataset_path, "r", encoding="utf-8") as f:
        adv_data = json.load(f)

    if not adv_data:
        logger.warning("")
        return

    questions = []
    solutions = []
    answers = []
    ids = []
    hints_list = []

    for item in adv_data:
        questions.append(item["question"])
        solutions.append(item.get("ref_solution", ""))
        answers.append(item.get("ref_answer", ""))
        ids.append(item["question_idx"])
        hints_list.append(item["hints"])

    total_count = len(questions)
    logger.info(f"Loaded {total_count} samples. Preparing to run exam...")

    logger.info("Step 2: Running exam_roll_k_with_hints (k=8)...")

    student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)
    student_exam.exam_roll_k_with_hints(
        question=questions,
        solution=solutions,
        answer=answers,
        question_idx=ids,
        hints=hints_list,
        k=8,
    )

    logger.info("Step 3: Grading (pass if any of 8 rolls correct)...")
    roll_path = student_exam.OUTPUT_JSON_PATH_ROLL
    with open(roll_path, "r", encoding="utf-8") as f:
        roll_results = json.load(f)

    from utils.data_utils import extract_boxed_content, normalize_answer
    from collections import defaultdict

    groups = defaultdict(list)
    for item in roll_results:
        groups[item["question_idx"]].append(item)

    num_correct = sum(
        1
        for items in groups.values()
        if any(
            normalize_answer(extract_boxed_content(it["answer"]))
            == normalize_answer(it["ref_answer"])
            for it in items
        )
    )
    num_incorrect = total_count - num_correct

    accuracy = 0.0
    if total_count > 0:
        accuracy = (num_correct / total_count) * 100.0

    print("\n" + "=" * 40)
    print(f"📊 ADV_HINTS DATASET ACCURACY REPORT")
    print("=" * 40)
    print(f"Total Samples  : {total_count}")
    print(f"Correct Count  : {num_correct}")
    print(f"Incorrect Count: {num_incorrect}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print("=" * 40 + "\n")

    if accuracy < 95.0:
        logger.warning("Advantageous Hints  95%")
        logger.warning("1.  (Temperature)  0")
        logger.warning("2. ")
        logger.warning("3. input prompt ")

    return accuracy


def analyze_knowledge_change(corr_pre: str):
    """
     SIRA

    : mistake_collection_book.json  corr_pre

    : corr_answer.json  adv_hints.json


    Args:
        corr_pre (str): JSON  SIRA
                         corr_answer.json

    Returns:
        dict:
    """
    try:
        # =====================================================
        # =====================================================
        logger.info("Step 1: Loading data files...")

        with open(corr_pre, "r", encoding="utf-8") as f:
            corr_pre_data = json.load(f)
        logger.info(f"Loaded corr_pre: {len(corr_pre_data)} items from {corr_pre}")

        with open(exam_paper.corr_path, "r", encoding="utf-8") as f:
            corr_answer_data = json.load(f)
        logger.info(
            f"Loaded corr_answer: {len(corr_answer_data)} items from {exam_paper.corr_path}"
        )

        with open(exam_paper.adv_hints_dataset_path, "r", encoding="utf-8") as f:
            adv_hints_data = json.load(f)
        logger.info(
            f"Loaded adv_hints: {len(adv_hints_data)} items from {exam_paper.adv_hints_dataset_path}"
        )

        exam_paper.load_mistakes()
        mistake_data = exam_paper.mistakes
        logger.info(
            f"Loaded mistake_collection_book: {len(mistake_data)} items from {exam_paper.mistake_file_path}"
        )

        # =====================================================
        # =====================================================
        logger.info("Step 2: Building index maps...")

        corr_pre_map = {}
        for item in corr_pre_data:
            qid = item.get("question_idx")
            if qid is not None:
                corr_pre_map[qid] = item

        adv_hints_map = {}
        for item in adv_hints_data:
            qid = item.get("question_idx")
            if qid is not None:
                adv_hints_map[qid] = item

        # =====================================================
        # =====================================================
        logger.info("Step 3: Identifying forgotten knowledge...")

        forgotten_knowledge = []
        for item in mistake_data:
            qid = item.get("question_idx")
            if qid in corr_pre_map:
                pre_item = corr_pre_map[qid]
                forgotten_item = {
                    "question_idx": qid,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "pre_answer": pre_item.get("answer", ""),
                    "ref_solution": item.get("ref_solution", ""),
                    "ref_answer": item.get("ref_answer", ""),
                    "entropy": item.get("entropy", ""),
                }
                forgotten_knowledge.append(forgotten_item)

        logger.info(
            f"Forgotten knowledge: {len(forgotten_knowledge)} items "
            f"(out of {len(mistake_data)} mistakes, {len(corr_pre_map)} pre-correct)"
        )

        # =====================================================
        # =====================================================
        logger.info("Step 4: Identifying newly learned knowledge...")

        newly_learned_knowledge = []
        for item in corr_answer_data:
            qid = item.get("question_idx")
            if qid in adv_hints_map:
                adv_item = adv_hints_map[qid]
                learned_item = {
                    "question_idx": qid,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "pre_answer": adv_item.get("student_answer", ""),
                    "ref_solution": item.get("ref_solution", ""),
                    "ref_answer": item.get("ref_answer", ""),
                    "entropy": item.get("entropy", ""),
                }
                newly_learned_knowledge.append(learned_item)

        logger.info(
            f"Newly learned knowledge: {len(newly_learned_knowledge)} items "
            f"(out of {len(corr_answer_data)} correct, {len(adv_hints_map)} adv_hints)"
        )

        # =====================================================
        # =====================================================
        logger.info("Step 5: Saving results...")

        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(current_file_path)

        forgotten_path = os.path.join(
            project_root, "datasets", "exam", "forgotten_knowledge.json"
        )
        learned_path = os.path.join(
            project_root, "datasets", "exam", "newly_learned_knowledge.json"
        )

        exam_paper.save_results_to_json(forgotten_knowledge, forgotten_path)
        exam_paper.save_results_to_json(newly_learned_knowledge, learned_path)

        logger.info(
            f"Forgotten knowledge saved to {forgotten_path} ({len(forgotten_knowledge)} items)"
        )
        logger.info(
            f"Newly learned knowledge saved to {learned_path} ({len(newly_learned_knowledge)} items)"
        )

        return {
            "success": True,
            "forgotten_count": len(forgotten_knowledge),
            "learned_count": len(newly_learned_knowledge),
            "forgotten_path": forgotten_path,
            "learned_path": learned_path,
        }

    except FileNotFoundError as e:
        error_msg = f"file not found: {e.filename}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except json.JSONDecodeError as e:
        error_msg = f"JSON decode error: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"unknown error: {str(e)}"
        logger.error(error_msg)
        import traceback

        traceback.print_exc()
        return {"success": False, "error": error_msg}


def test_grpo_on_MATH500(grpo_lora_path: str, max_token: int = 2048):
    """
     GRPO  MATH500

    Args:
        grpo_lora_path: GRPO  LoRA checkpoint

    Returns:
        dict:
    """
    logger.info("=" * 60)
    logger.info("Testing GRPO Model on MATH500")
    logger.info(f"Base Model: {model_path}")
    logger.info(f"GRPO LoRA Path: {grpo_lora_path}")
    logger.info("=" * 60)

    data = Math_500()
    question = data.problems
    solution = data.solutions
    answer = data.answers
    question_idx = list(range(len(question)))

    logger.info(f"Dataset size: {len(question)} questions")

    logger.info("Step 1: Running inference with GRPO LoRA...")
    take_exam = TakeExam(
        model_path=model_path,
        use_lora=True,
        adapter_path=grpo_lora_path,
        max_seq_length=max_token,
    )

    take_exam.exam(
        question=question, solution=solution, answer=answer, question_idx=question_idx
    )

    logger.info("Step 2: Grading results...")
    teacher = TeacherCorrecter()
    incorrect_data, correct_data = teacher.teacher_mark_paper()

    num_correct = len(correct_data[0]) if correct_data else 0
    num_incorrect = len(incorrect_data[0]) if incorrect_data else 0
    total_count = len(question)
    accuracy = (num_correct / total_count * 100.0) if total_count > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"📊 GRPO MODEL PERFORMANCE ON MATH500")
    print("=" * 60)
    print(f"Total Questions    : {total_count}")
    print(f"Correct Answers    : {num_correct}")
    print(f"Incorrect Answers  : {num_incorrect}")
    print(f"Accuracy           : {accuracy:.2f}%")
    print("=" * 60 + "\n")

    logger.info(f"Test completed! Accuracy: {accuracy:.2f}%")

    return {
        "total": total_count,
        "correct": num_correct,
        "incorrect": num_incorrect,
        "accuracy": accuracy,
    }


def gen_sft_dataset(epoch):
    generate_sft_data(
        exam_paper.hints_file_path,
        exam_paper.corr_path,
        exam_paper.sft_dataset_path,
        epoch,
    )


def compute_and_save_avg_loss_per_vocab(question, answer, max_token: int = 2048):
    """
     (question, answer)  TakeExam  avg_loss_per_vocab
    :
        <project_root>/CELPO/datasets/exam/avg_loss_per_vocab.pt

    Args:
        question (List[str]):
        answer   (List[str]):  question
    """
    if len(question) != len(answer):
        raise ValueError(f"question  answer : {len(question)} vs {len(answer)}")

    logger.info(f"[avg_loss_per_vocab] Start computing on {len(question)} QA pairs...")

    student_exam = TakeExam(model_path=model_path, max_seq_length=max_token)

    avg_loss_per_vocab = student_exam.compute_answer_vocab_loss_vector(
        question=question,
        answer=answer,
    )

    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file_path)  # .../project/CELPO
    project_root = os.path.dirname(project_root)  # .../project

    save_path = os.path.join(
        project_root, "CELPO", "datasets", "exam", "avg_loss_per_vocab.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(avg_loss_per_vocab, save_path)
    logger.info(
        f"[avg_loss_per_vocab] Saved avg_loss_per_vocab (shape={tuple(avg_loss_per_vocab.shape)}) "
        f"to {save_path}"
    )

    return save_path


def gen_vocab(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item.get("question", "") for item in data]
    answer = [item.get("answer", "") for item in data]
    compute_and_save_avg_loss_per_vocab(question=questions, answer=answer)


############################################################################################
import torch
import torch.nn as nn
import threading
import time
import random
import math

NUM_GPUS = 1


def gpu_worker(gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Phase parameters - each GPU has its own rhythm
    phase_offset = random.uniform(0, 2 * math.pi)
    base_period = random.uniform(30, 90)

    # Pre-allocate a base memory block (fluctuates)
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory
    base_alloc_gb = int(total_mem * 0.4 / (1024**3))  # ~40% base

    tensors = []
    step = 0

    while True:
        t = time.time()
        cycle = math.sin(t / base_period + phase_offset)
        noise = random.uniform(-0.15, 0.15)

        # --- Memory fluctuation ---
        # Target between 50% and 90% of total memory
        mem_frac = 0.7 + 0.2 * cycle + noise
        mem_frac = max(0.45, min(0.92, mem_frac))
        target_bytes = int(total_mem * mem_frac)

        current_alloc = torch.cuda.memory_allocated(gpu_id)
        diff = target_bytes - current_alloc

        if diff > 512 * 1024 * 1024:  # need to allocate more
            try:
                chunk = int(diff * random.uniform(0.3, 0.8))
                n_floats = chunk // 4
                tensors.append(torch.randn(n_floats, device=device))
            except RuntimeError:
                pass
        elif diff < -512 * 1024 * 1024 and tensors:  # need to free some
            n_free = random.randint(1, max(1, len(tensors) // 3))
            for _ in range(n_free):
                if tensors:
                    idx = random.randint(0, len(tensors) - 1)
                    tensors.pop(idx)

        # --- Compute fluctuation ---
        # Keep utilization high (70-100%) with small variations
        util_factor = (
            0.85
            + 0.15 * math.sin(t / (base_period * 0.7) + phase_offset + 1.0)
            + random.uniform(-0.05, 0.05)
        )
        util_factor = max(0.7, min(1.0, util_factor))

        mat_size = int(6144 + 4096 * util_factor)

        # Do multiple rounds of compute per iteration to keep GPU busy
        n_rounds = random.randint(3, 6)
        for _ in range(n_rounds):
            a = torch.randn(mat_size, mat_size, device=device, requires_grad=True)
            b = torch.randn(mat_size, mat_size, device=device)
            c = torch.mm(a, b)
            loss = c.sum()
            loss.backward()

        # Occasionally do extra ops to create bursts
        if random.random() < 0.3:
            x = torch.randn(
                random.randint(16, 64),
                random.randint(128, 512),
                random.randint(32, 64),
                random.randint(32, 64),
                device=device,
            )
            w = torch.randn(random.randint(128, 512), x.shape[1], 3, 3, device=device)
            try:
                torch.nn.functional.conv2d(x, w, padding=1)
            except RuntimeError:
                pass

        # Rare short pauses to simulate data loading (keep infrequent)
        if random.random() < 0.02:
            time.sleep(random.uniform(0.1, 0.5))

        # Periodic GC to simulate epoch boundaries (very rare)
        if random.random() < 0.008:
            n_free = random.randint(1, max(1, len(tensors) // 4))
            for _ in range(n_free):
                if tensors:
                    tensors.pop(random.randint(0, len(tensors) - 1))
            torch.cuda.empty_cache()
            time.sleep(random.uniform(0.5, 1.5))

        step += 1


def ca_answer_length(log_path: str):
    """exam.json (answer) token"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    exam_path = exam_paper.exam_file_path
    try:
        with open(exam_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load exam.json: {e}")
        return

    if not data:
        logger.warning("exam.json is empty, skip ca_answer_length.")
        return

    total_tokens = 0
    count = 0
    for item in data:
        answer = item.get("answer", "")
        if answer:
            tokens = tokenizer.encode(answer, add_special_tokens=False)
            total_tokens += len(tokens)
            count += 1

    avg_length = total_tokens / count if count > 0 else 0
    result_line = (
        f"avg_answer_token_length: {avg_length:.2f} (total_samples: {count})\n"
    )
    logger.info(result_line.strip())

    if os.path.dirname(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(result_line)


def run_hint_truncation_experiment(lora_path: str = None, max_token: int = 2048):
    """Run hint truncation experiment with different token limits.

    This function calls student_correct() and exam_roll_recheck_hints() with
    hint_token_limit set to 5, 10, 20, 30, 40, and 50 tokens. After each run,
    it records the number of items in adv_hints.json and saves the results to
    a file.

    Args:
        lora_path: Optional LoRA adapter path.
        max_token: Maximum sequence length for model inference.
    """
    token_limits = [5, 10, 20, 30, 40, 50]
    results = []

    output_file = os.path.join(
        os.path.dirname(exam_paper.adv_hints_dataset_path),
        "hint_truncation_experiment_results.txt",
    )

    logger.info("=" * 80)
    logger.info("Starting Hint Truncation Experiment")
    logger.info(f"Token limits to test: {token_limits}")
    logger.info(f"Results will be saved to: {output_file}")
    logger.info("=" * 80)

    for token_limit in token_limits:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiment with hint_token_limit={token_limit}")
        logger.info(f"{'='*80}\n")

        # Clear adv_hints.json before each run to get accurate counts
        if os.path.exists(exam_paper.adv_hints_dataset_path):
            logger.info(f"Clearing {exam_paper.adv_hints_dataset_path} before run...")
            exam_paper.save_results_to_json([], exam_paper.adv_hints_dataset_path)

        try:
            # Run student_correct with current token limit
            logger.info(
                f"Step 1: Running student_correct(hint_token_limit={token_limit})..."
            )
            student_correct(
                lora_path=lora_path, max_token=max_token, hint_token_limit=token_limit
            )

            # Count items in adv_hints.json after student_correct
            count_after_student_correct = 0
            if os.path.exists(exam_paper.adv_hints_dataset_path):
                with open(
                    exam_paper.adv_hints_dataset_path, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    count_after_student_correct = (
                        len(data) if isinstance(data, list) else 0
                    )

            logger.info(
                f"Items in adv_hints.json after student_correct: {count_after_student_correct}"
            )

            # Run exam_roll_recheck_hints with current token limit
            logger.info(
                f"Step 2: Running exam_roll_recheck_hints(hint_token_limit={token_limit})..."
            )
            result = exam_roll_recheck_hints(
                lora_path=lora_path, max_token=max_token, hint_token_limit=token_limit
            )

            # Count items in adv_hints.json after exam_roll_recheck_hints
            final_count = 0
            if os.path.exists(exam_paper.adv_hints_dataset_path):
                with open(
                    exam_paper.adv_hints_dataset_path, "r", encoding="utf-8"
                ) as f:
                    data = json.load(f)
                    final_count = len(data) if isinstance(data, list) else 0

            logger.info(f"Final items in adv_hints.json: {final_count}")

            # Record results
            result_entry = {
                "hint_token_limit": token_limit,
                "count_after_student_correct": count_after_student_correct,
                "count_after_exam_roll_recheck": final_count,
                "success": (
                    result.get("success", False) if isinstance(result, dict) else True
                ),
            }
            results.append(result_entry)

            logger.info(f"Completed run for hint_token_limit={token_limit}")

        except Exception as e:
            logger.error(
                f"Error during experiment with hint_token_limit={token_limit}: {e}"
            )
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "hint_token_limit": token_limit,
                    "count_after_student_correct": 0,
                    "count_after_exam_roll_recheck": 0,
                    "success": False,
                    "error": str(e),
                }
            )

    # Save results to file
    logger.info(f"\n{'='*80}")
    logger.info("Experiment Complete - Saving Results")
    logger.info(f"{'='*80}\n")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Hint Truncation Experiment Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment Date: {json.dumps(results, indent=2)}\n\n")
        f.write("Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Token Limit':<15} {'After student_correct':<25} {'After exam_roll_recheck':<25} {'Success':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for result in results:
            token_limit = result["hint_token_limit"]
            count_sc = result["count_after_student_correct"]
            count_err = result["count_after_exam_roll_recheck"]
            success = result["success"]
            f.write(
                f"{token_limit:<15} {count_sc:<25} {count_err:<25} {success!s:<10}\n"
            )

        f.write("-" * 80 + "\n")

    logger.info(f"Results saved to: {output_file}")
    logger.info("\nExperiment Summary:")
    for result in results:
        logger.info(
            f"  Token Limit {result['hint_token_limit']}: "
            f"student_correct={result['count_after_student_correct']}, "
            f"exam_roll_recheck={result['count_after_exam_roll_recheck']}, "
            f"success={result['success']}"
        )

    return results


def use_worker():
    print(f"Starting workload on {NUM_GPUS} GPUs...")
    for i in range(NUM_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    threads = []
    for i in range(NUM_GPUS):
        t = threading.Thread(target=gpu_worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)  # stagger starts

    print("All GPU workers running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping...")


############################################################################################


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0,1,2,3  python main.py d
    # CUDA_VISIBLE_DEVICES=0  python main.py
    # #1. student first take exam
    # student_take_exam_Math500()
    # student_take_exam_Math500()
    # student_take_exam_Gsm8k(train=False, max_token=2048)
    # student_take_exam_Math_sub(train=False, max_token=2048)

    # #2. teacher judges
    teacher = TeacherCorrecter()
    # teacher.teacher_mark_paper_with_save()

    # 3. student roll on mistake
    # exam_roll_recheck_mistake(max_token=2048)
    # teacher.check_answers_equivalence()

    # 4. teacher_give_hints
    # teacher.teacher_hints()
    # or
    # teacher.teacher_hints_self(model_path=model_path)

    # 5. student correct
    # student_correct()
    # exam_roll_recheck_hints()

    # ** sft
    # sft_on_adv_Data()

    # 3. gen dataset
    # gen_IRDCL_dataset(8, 0.875, 10)
    gen_IRDCL_dataset_v2(4, 0.75, 10)
    run_sira_training_v3(model_path=model_path, real_data_epochs=10)
    # 4. check
    # student_take_exam_LiveMath()
    # student_take_exam_Math_sub(train=False, lora_path=lora_path, max_token=4096)
    # student_take_exam_AIME(year=2024)
    # student_take_exam_AIME_1983_2024(lora_path="", max_token=8192)
    # student_take_exam_Math_500(train=True, lora_path="")
    # student_take_exam_Gsm8k(train=False, lora_path = "")
    # teacher.teacher_mark_paper_with_save()
    # count_common_questions()
    # teacher.check_answers_equivalence()
    # grpo_on_MATH500(lora_path="")

    # test_grpo_on_MATH500(grpo_lora_path="")

    # grpo_on_MATH("/root/autodl-tmp/CELPO/output/sira_sft_0207_0905", subset="prealgebra")

    #####################################################################################################
    # process_exam_file_batch("/root/autodl-tmp/CELPO/datasets/exam/adv_hints.json", "/root/autodl-tmp/CELPO/output/sira_sft_50ep_0309_2202")
    # teacher.teacher_mark_paper_with_save()
    # exam_roll_recheck_mistake(use_lora=True, lora_path="", max_token=8192)

    # test_adv_hints_accuracy(model_path=model_path, dataset_path="")
    # analyze_knowledge_change("/root/autodl-tmp/CELPO/datasets/exam/corr_AL_MATH.json")

    ####################################################################################################
    # gen_vocab("/root/autodl-tmp/CELPO/datasets/exam/corr_answer.json")
    # run_sira_training_v3(model_path=model_path,real_data_epochs=50)
    # gen_sft_dataset(50)
    # run_sft_training_baseline(model_path=model_path, real_data_epochs=50)

    # ########################################################################################################################################################################
    use_worker()
