import re
from typing import Optional
import prompt
import os
import json
import random


def extract_KNOWN(text: str) -> Optional[str]:
    pattern = r'<KNOWN>\s*(.*?)\s*</KNOWN>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_hints(text: str) -> Optional[str]:
    pattern = r'<hints>\s*(.*?)\s*</hints>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_thinking(text: str) -> Optional[str]:
    pattern = r'<thinking>\s*(.*?)\s*</thinking>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_conclusion(text: str) -> Optional[str]:
    pattern = r'<conclusion>\s*(.*?)\s*</conclusion>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_reason(text: str) -> Optional[str]:
    pattern = r'<reason>\s*(.*?)\s*</reason>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None

def extract_boxed_content(text: str) -> Optional[str]:
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[-1].strip()
    return ""

def normalize_answer(answer: str) -> str:
    if answer is None: return ""
    answer = answer.replace(" ", "").lower()
    answer = re.sub(r'\\[a-zA-Z]+', '', answer) 
    answer = re.sub(r'[^0-9a-zA-Z\+\-\*/=\.\,]', '', answer)
    return answer

def collate_fn(batch):
    return {
            'prompts': [item['prompt'] for item in batch],
            'reference_answers': [item['reference_answer'] for item in batch]
            }


def remove_null_hints(file_path):
    if not os.path.exists(file_path):
        print(f"erro, can not find: {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_count = len(data)

        filtered_data = [item for item in data if item.get('hints') is not None]
        
        new_count = len(filtered_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)

        print(f"finished: {file_path}")
        print(f"from: {original_count}")
        print(f"to: {new_count}")
        print(f"deleted: {original_count - new_count}")

    except Exception as e:
        print(f"error: {e}")



def filter_json_by_question_idx(exam_path, hints_exam_result_path):
    try:
        with open(exam_path, 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        
        with open(hints_exam_result_path, 'r', encoding='utf-8') as f:
            data_b = json.load(f)
        
        question_idx_in_b = set()
        for item in data_b:
            if 'question_idx' in item:
                question_idx_in_b.add(item['question_idx'])
        
        data_c = []
        removed_count = 0
        for item in data_a:
            if 'question_idx' in item:
                if item['question_idx'] not in question_idx_in_b:
                    data_c.append(item)
                else:
                    removed_count += 1
            else:
                data_c.append(item)
        
        with open(corr_path, 'w', encoding='utf-8') as f:
            json.dump(data_c, f, ensure_ascii=False, indent=2)
        
        result = {
            'success': True,
            'original_count': len(data_a),
            'question_idx_in_b': len(question_idx_in_b),
            'remaining_count': len(data_c),
            'removed_count': removed_count,
            'message': f'finished! from{len(data_a)}del{removed_count}items，remains{len(data_c)}items to{corr_path}'
        }
        
        print(result['message'])
        return result
        
    except FileNotFoundError as e:
        error_msg = f'file not found: {e.filename}'
        print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    
    except json.JSONDecodeError as e:
        error_msg = f'JSON decode error: {str(e)}'
        print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    
    except Exception as e:
        error_msg = f'unknown error: {str(e)}'
        print(error_msg)
        return {
            'success': False,
            'error': error_msg
        }
    


def generate_irdcl_dataset(corr_path, adv_hints_path, disadv_hints_path, output_path, batch_size, anchor_k=0.5):
    """
    # Construct IRDCL training data.
    # Logic: Split the Hints data into chunks, and pair each chunk with randomly sampled Corr data to form a batch.
    # 
    # Args:
    #     anchor_k: The proportion of anchor (corr) data in each batch.
    """
    std_num_anchors = int(batch_size * anchor_k)
    std_num_hints = batch_size - std_num_anchors
    
    if std_num_hints <= 0:
        raise ValueError(f"Batch size {batch_size} implies 0 hints with anchor_k={anchor_k}")

    print(f"Standard Batch Config: Total={batch_size} | Hints={std_num_hints} | Anchors={std_num_anchors}")

    print("Loading data...")
    with open(corr_path, 'r', encoding='utf-8') as f:
        corr_data = json.load(f)
    
    with open(adv_hints_path, 'r', encoding='utf-8') as f:
        adv_data = json.load(f)
        
    with open(disadv_hints_path, 'r', encoding='utf-8') as f:
        disadv_data = json.load(f)

    combined_hints_data = adv_data + disadv_data
    total_hints = len(combined_hints_data)
    print(f"Data Loaded. Hints: {total_hints}, Anchors Pool: {len(corr_data)}")

    final_dataset = []

    random.shuffle(combined_hints_data)

    for i in range(0, total_hints, std_num_hints):
        current_batch = []
        hints_chunk = combined_hints_data[i : i + std_num_hints]
        actual_hint_count = len(hints_chunk)
        
        for item in hints_chunk:
            entry = {
                "question_idx": item.get("question_idx"),
                "question": item.get("question"),
                "answer": item.get("student_answer"),
                "ref_answer": item.get("ref_answer"),
                "ref_solution": item.get("ref_solution"),
                "hints": item.get("hints"),
                "type": "hint_data" 
            }
            current_batch.append(entry)
        
        if anchor_k >= 1.0 or anchor_k <= 0.0:
            needed_anchor_count = 0 if anchor_k <= 0 else actual_hint_count
        else:
            ratio_factor = anchor_k / (1.0 - anchor_k)
            needed_anchor_count = int(round(actual_hint_count * ratio_factor))
        
        anchors_chunk = random.choices(corr_data, k=needed_anchor_count)

        for item in anchors_chunk:
            entry = {
                "question_idx": item.get("question_idx"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "ref_answer": item.get("ref_answer"),
                "ref_solution": item.get("ref_solution"),
                "hints": None,
                "type": "anchor_data"
            }
            current_batch.append(entry)
        
        random.shuffle(current_batch)
        
        final_dataset.extend(current_batch)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"Done. Generated {len(final_dataset)} items.")
    print(f"Saved to {output_path}")
