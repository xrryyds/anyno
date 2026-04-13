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
    """
    提取 LaTeX 字符串中最后一个 \boxed{...} 里的内容。
    支持嵌套括号，例如 \boxed{\frac{1}{2}}。
    """
    if not text: return ""
    
    # 找到最后一个 \boxed{ 的位置
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""

    # 移动索引到 \boxed{ 之后
    i = idx + 7 
    content_start = i
    brace_balance = 0 
    
    # 开始遍历字符
    while i < len(text):
        char = text[i]
        
        if char == '{':
            brace_balance += 1
        elif char == '}':
            if brace_balance == 0:
                return text[content_start:i].strip()
            else:
                brace_balance -= 1
        
        i += 1
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
    


def generate_irdcl_dataset_syn(corr_path, adv_hints_path, disadv_hints_path, output_path, batch_size, anchor_k=0.5):
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
        
    # with open(disadv_hints_path, 'r', encoding='utf-8') as f:
    #     disadv_data = json.load(f)

    # combined_hints_data = adv_data + disadv_data
    combined_hints_data = adv_data
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



def generate_irdcl_datase_v2(corr_path, adv_hints_path, disadv_hints_path, output_path, batch_size, anchor_k=0.5, epoch=3):
    """
    Construct IRDCL training data (v2).
    Logic: Same as generate_irdcl_dataset, but anchor pool uses cross-epoch
    non-replacement sampling. The anchor pool is shuffled once at the start,
    and a global index advances across all epochs without resetting.
    When the pool is exhausted, it reshuffles and restarts.
    """
    # 计算标准的 batch 分配
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
        
    # with open(disadv_hints_path, 'r', encoding='utf-8') as f:
    #     disadv_data = json.load(f)

    # combined_hints_data = adv_data + disadv_data
    combined_hints_data = adv_data
    total_hints = len(combined_hints_data)
    print(f"Data Loaded. Hints: {total_hints}, Anchors Pool: {len(corr_data)}")

    final_dataset = []

    # 全 epoch 共享的 anchor 池：初始打乱一次，全局索引跨 epoch 不重置
    shuffled_anchors = corr_data.copy()
    random.shuffle(shuffled_anchors)
    anchor_pool_idx = 0

    # --- 开始 Epoch 循环 ---
    for ep in range(epoch):
        print(f"Processing Epoch {ep + 1}/{epoch}...")
        
        # 每个 epoch 开始前打乱 hints 数据的顺序
        # 这样每个 epoch 的 batch 组合都会不同
        random.shuffle(combined_hints_data)

        # 遍历 hints 数据生成 batch
        for i in range(0, total_hints, std_num_hints):
            current_batch = []
            
            # 切片获取当前的 hints 块
            hints_chunk = combined_hints_data[i : i + std_num_hints]
            actual_hint_count = len(hints_chunk)
            
            # 添加 hints 数据
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
            
            # 计算需要补充的 anchor 数量
            # 如果是最后一个 batch，hints 数量可能少于 std_num_hints，所以按比例重新计算
            if anchor_k >= 1.0 or anchor_k <= 0.0:
                needed_anchor_count = 0 if anchor_k <= 0 else actual_hint_count
            else:
                ratio_factor = anchor_k / (1.0 - anchor_k)
                needed_anchor_count = int(round(actual_hint_count * ratio_factor))
            
            # 跨 epoch 不放回抽样 anchor 数据
            # 当池用完时，重新打乱并从头开始
            anchors_chunk = []
            for _ in range(needed_anchor_count):
                if anchor_pool_idx >= len(shuffled_anchors):
                    # 池耗尽，重新打乱并重置索引
                    random.shuffle(shuffled_anchors)
                    anchor_pool_idx = 0
                anchors_chunk.append(shuffled_anchors[anchor_pool_idx])
                anchor_pool_idx += 1

            # 添加 anchor 数据
            for item in anchors_chunk:
                entry = {
                    "question_idx": item.get("question_idx"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "ref_answer": item.get("ref_answer"),
                    "ref_solution": item.get("ref_solution"),
                    "hints": None,
                    "type": "anchor_data",
                    "ref_beta": item.get("ref_beta"),
                }
                current_batch.append(entry)
            
            # 打乱当前 batch 内部顺序
            random.shuffle(current_batch)
            
            # 将当前 batch 加入总数据集
            final_dataset.extend(current_batch)

    # --- 保存结果 ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"Done. Generated total {len(final_dataset)} items over {epoch} epochs.")
    print(f"Saved to {output_path}")



def generate_irdcl_dataset(corr_path, adv_hints_path, disadv_hints_path, output_path, batch_size, anchor_k=0.5, epoch=3):
    """
    # Construct IRDCL training data.
    # Logic: Repeat the generation process for 'epoch' times.
    # In each epoch, shuffle hints, split into chunks, and pair with randomly sampled Corr data.
    """
    # 计算标准的 batch 分配
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

    # with open(disadv_hints_path, 'r', encoding='utf-8') as f:
    #     disadv_data = json.load(f)

    # combined_hints_data = adv_data + disadv_data
    combined_hints_data = adv_data
    total_hints = len(combined_hints_data)
    print(f"Data Loaded. Hints: {total_hints}, Anchors Pool: {len(corr_data)}")

    final_dataset = []

    # --- 开始 Epoch 循环 ---
    for ep in range(epoch):
        print(f"Processing Epoch {ep + 1}/{epoch}...")

        # 每个 epoch 开始前打乱 hints 数据的顺序
        # 这样每个 epoch 的 batch 组合都会不同
        random.shuffle(combined_hints_data)

        # 每个 epoch 开始前打乱 anchor 数据池，用于无放回抽样
        shuffled_anchors = corr_data.copy()
        random.shuffle(shuffled_anchors)
        anchor_pool_idx = 0

        # 遍历 hints 数据生成 batch
        for i in range(0, total_hints, std_num_hints):
            current_batch = []

            # 切片获取当前的 hints 块
            hints_chunk = combined_hints_data[i : i + std_num_hints]
            actual_hint_count = len(hints_chunk)

            # 添加 hints 数据
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

            # 计算需要补充的 anchor 数量
            # 如果是最后一个 batch，hints 数量可能少于 std_num_hints，所以按比例重新计算
            if anchor_k >= 1.0 or anchor_k <= 0.0:
                needed_anchor_count = 0 if anchor_k <= 0 else actual_hint_count
            else:
                ratio_factor = anchor_k / (1.0 - anchor_k)
                needed_anchor_count = int(round(actual_hint_count * ratio_factor))

            # 无放回抽样 anchor 数据，如果不够则循环使用
            anchors_chunk = []
            for _ in range(needed_anchor_count):
                anchors_chunk.append(shuffled_anchors[anchor_pool_idx % len(corr_data)])
                anchor_pool_idx += 1

            # 添加 anchor 数据
            for item in anchors_chunk:
                entry = {
                    "question_idx": item.get("question_idx"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "ref_answer": item.get("ref_answer"),
                    "ref_solution": item.get("ref_solution"),
                    "hints": None,
                    "type": "anchor_data",
                    "ref_beta": item.get("ref_beta"),
                }
                current_batch.append(entry)

            # 打乱当前 batch 内部顺序
            random.shuffle(current_batch)

            # 将当前 batch 加入总数据集
            final_dataset.extend(current_batch)

    # --- 保存结果 ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print(f"Done. Generated total {len(final_dataset)} items over {epoch} epochs.")
    print(f"Saved to {output_path}")



def generate_sft_data(adv_hints_path: str, corr_path: str, output_path: str, epoch: int):
    """Generate SFT data from adv_hints and corr_answer.

    - From adv_hints: question -> question, ref_solution -> answer
    - From corr_answer: question -> question, answer -> answer
    - Repeat for `epoch` times; each epoch shuffles internally before appending.
    """
    try:
        # 1. load adv_hints
        with open(adv_hints_path, 'r', encoding='utf-8') as f:
            adv_data = json.load(f)

        if not isinstance(adv_data, list):
            print(f"[generate_sft_data] adv_hints is not a list: {adv_hints_path}")
            return

        adv_samples = []
        for item in adv_data:
            adv_samples.append({
                "question": item.get("question", ""),
                "answer": item.get("ref_solution", ""),
            })

        # 2. load corr_answer
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_data = json.load(f)

        if not isinstance(corr_data, list):
            print(f"[generate_sft_data] corr_answer is not a list: {corr_path}")
            return

        corr_samples = []
        for item in corr_data:
            corr_samples.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            })

        base_samples = adv_samples + corr_samples
        print(f"[generate_sft_data] base_samples: adv={len(adv_samples)}, corr={len(corr_samples)}, total={len(base_samples)}")

        all_samples = []
        for ep in range(epoch):
            epoch_samples = base_samples.copy()
            random.shuffle(epoch_samples)
            all_samples.extend(epoch_samples)
            print(f"[generate_sft_data] epoch {ep+1}/{epoch} appended {len(epoch_samples)} samples")

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)

        print(f"[generate_sft_data] Done. Generated {len(all_samples)} items, saved to {output_path}")

    except FileNotFoundError as e:
        print(f"[generate_sft_data] file not found: {e.filename}")
    except json.JSONDecodeError as e:
        print(f"[generate_sft_data] JSON decode error: {e}")
    except Exception as e:
        print(f"[generate_sft_data] unknown error: {e}")
    




def remove_null_hints(file_path):
    if not os.path.exists(file_path):
        print(f"erro, can not find: {file_path}")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("error: JSON not (List)。")
            return

        original_count = len(data)

        cleaned_data = [item for item in data if item.get("hints") is not None]

        filtered_count = len(cleaned_data)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

        print(f"finished！")
        print(f"文件路径: {file_path}")
        print(f"原始数据条数: {original_count}")
        print(f"剩余数据条数: {filtered_count}")
        print(f"删除了 {original_count - filtered_count} 条数据。")

    except json.JSONDecodeError:
        print("错误: 文件格式不是有效的 JSON。")
    except Exception as e:
        print(f"发生未知错误: {e}")