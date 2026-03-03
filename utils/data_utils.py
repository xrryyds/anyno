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
    Construct IRDCL training data.
    - If anchor_k == 0: Pure Mode B (Hint) training, no anchors.
    - If anchor_k > 0: Interleaved training (Hint, Anchor, Hint, Anchor...).
    """
    
    print(f"Generating Dataset with anchor_k={anchor_k}...")

    # --- 1. 加载 Hint 数据 (Mode B) ---
    # 这里假设 adv_hints 和 disadv_hints 都属于 Hint 数据，如果有区分需求请自行筛选
    if os.path.exists(adv_hints_path):
        with open(adv_hints_path, 'r', encoding='utf-8') as f:
            adv_data = json.load(f)
    else:
        adv_data = []
        print(f"Warning: {adv_hints_path} not found.")

    # 如果需要合并 disadv_hints_path，可以在这里加载并 extend
    # combined_hints_data = adv_data + disadv_data 
    combined_hints_data = adv_data 
    total_hints = len(combined_hints_data)
    print(f"Loaded {total_hints} Hint samples.")

    final_dataset = []
    
    # --- 2. 纯 Mode B 模式 (anchor_k == 0) ---
    if anchor_k == 0:
        print(">>> Mode: Pure Mode B (No Anchors)")
        
        for ep in range(epoch):
            print(f"Processing Epoch {ep + 1}/{epoch}...")
            # 每个 epoch 打乱一次数据
            random.shuffle(combined_hints_data)
            
            for item in combined_hints_data:
                final_dataset.append({
                    "question_idx": item.get("question_idx"),
                    "question": item.get("question"),
                    # 注意：SIRA Trainer 中通常期望 key 为 'hints' 和 'answer'
                    "hints": item.get("hints"),
                    "answer": item.get("student_answer") or item.get("ref_answer"), # 优先用学生答案，或者参考答案
                    "ref_answer": item.get("ref_answer"),
                    "ref_solution": item.get("ref_solution"),
                    "type": "mode_b_generation"  # 标记为 Mode B
                })

    # --- 3. 混合穿插模式 (anchor_k > 0) ---
    else:
        print(f">>> Mode: Interleaved (Hint + Anchor), Ratio ~ {anchor_k}")
        
        # 3.1 加载 Anchor 数据
        with open(corr_path, 'r', encoding='utf-8') as f:
            corr_data = json.load(f)
        
        # 3.2 计算 Batch 分配
        std_num_anchors = int(batch_size * anchor_k)
        std_num_hints = batch_size - std_num_anchors
        
        if std_num_hints <= 0:
            raise ValueError(f"Batch size {batch_size} implies 0 hints with anchor_k={anchor_k}")

        # 3.3 Anchor 池扩充逻辑
        total_batches = (total_hints + std_num_hints - 1) // std_num_hints
        approx_anchors_per_epoch = total_batches * std_num_anchors
        total_anchors_needed = approx_anchors_per_epoch * epoch
        
        repeat_factor = max(1, (total_anchors_needed + len(corr_data) - 1) // len(corr_data))
        extended_corr_data = []
        for i in range(repeat_factor):
            shuffled_copy = corr_data.copy()
            random.shuffle(shuffled_copy)
            extended_corr_data.extend(shuffled_copy)
        
        print(f"Extended anchor pool size: {len(extended_corr_data)}")
        
        anchor_idx = 0

        # 3.4 Epoch 循环
        for ep in range(epoch):
            print(f"Processing Epoch {ep + 1}/{epoch}...")
            random.shuffle(combined_hints_data)
            
            # 按 Batch 步长遍历 Hints
            for i in range(0, total_hints, std_num_hints):
                
                # A. 获取 Hints Chunk
                hints_chunk = combined_hints_data[i : i + std_num_hints]
                actual_hint_count = len(hints_chunk)
                
                # B. 获取 Anchors Chunk
                ratio_factor = anchor_k / (1.0 - anchor_k)
                needed_anchor_count = int(round(actual_hint_count * ratio_factor))
                
                # 边界保护
                if anchor_idx + needed_anchor_count > len(extended_corr_data):
                    # 如果不够了，从头再取（循环利用）
                    anchor_idx = 0
                
                anchors_chunk = extended_corr_data[anchor_idx : anchor_idx + needed_anchor_count]
                anchor_idx += needed_anchor_count

                # C. 交替穿插 (Zip)
                min_len = min(len(hints_chunk), len(anchors_chunk))
                
                for k in range(min_len):
                    h_item = hints_chunk[k]
                    a_item = anchors_chunk[k]
                    
                    # 添加 Hint (Mode B)
                    final_dataset.append({
                        "question_idx": h_item.get("question_idx"),
                        "question": h_item.get("question"),
                        "hints": h_item.get("hints"),
                        "answer": h_item.get("student_answer") or h_item.get("ref_answer"),
                        "ref_answer": h_item.get("ref_answer"),
                        "ref_solution": h_item.get("ref_solution"),
                        "type": "mode_b_generation"
                    })
                    
                    # 添加 Anchor (Pure SFT)
                    final_dataset.append({
                        "question_idx": a_item.get("question_idx"),
                        "question": a_item.get("question"),
                        "hints": None, # Anchor 没有 Hints
                        "answer": a_item.get("answer") or a_item.get("ref_answer"),
                        "ref_answer": a_item.get("ref_answer"),
                        "ref_solution": a_item.get("ref_solution"),
                        "type": "anchor_data",
                        "ref_beta": a_item.get("ref_beta"),
                    })

                # D. 处理多出来的部分 (Tail)
                # 只有当 hints 和 anchors 数量不对等时才会执行这里
                if len(hints_chunk) > min_len:
                    for h_item in hints_chunk[min_len:]:
                        final_dataset.append({
                            "question_idx": h_item.get("question_idx"),
                            "question": h_item.get("question"),
                            "hints": h_item.get("hints"),
                            "answer": h_item.get("student_answer") or h_item.get("ref_answer"),
                            "ref_answer": h_item.get("ref_answer"),
                            "ref_solution": h_item.get("ref_solution"),
                            "type": "mode_b_generation"
                        })
                
                if len(anchors_chunk) > min_len:
                    for a_item in anchors_chunk[min_len:]:
                        final_dataset.append({
                            "question_idx": a_item.get("question_idx"),
                            "question": a_item.get("question"),
                            "hints": None,
                            "answer": a_item.get("answer") or a_item.get("ref_answer"),
                            "ref_answer": a_item.get("ref_answer"),
                            "ref_solution": a_item.get("ref_solution"),
                            "type": "anchor_data",
                            "ref_beta": a_item.get("ref_beta"),
                        })

    # --- 4. 保存 ---
    print(f"\nSaved {len(final_dataset)} items to {output_path}")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=4)

    print("Dataset generation complete.")


# 使用示例
# generate_irdcl_dataset_interleaved(..., batch_size=32, anchor_k=0.5, ...)



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