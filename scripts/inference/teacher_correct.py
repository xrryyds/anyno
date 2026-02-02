from utils import FileIOUtils, extract_hints ,extract_boxed_content, normalize_answer
from openai import OpenAI, RateLimitError, APIError
from prompt.prompts import TEACHER_CORRECT_PROMPT, OREAL_CORRECT_PROMPT
import time

base_url = "https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps"
api_key = "k1y21hll8l0eurf7t3dg4enb56g0hhjjszf4"

class TeacherCorrecter:
    def __init__(self):
        self.file = FileIOUtils()
        self.acc = 0
        self.err_count = 0
        self.toolong_count = 0
        self.acc_count = 0

    def teacher_hints(self) -> bool:
        print("Starting teacher hinting...")
        print("load mistakes...")
        self.file.load_mistakes()
        m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution, m_entropy = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(m_question))


        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        h_question_idx = []

        print(f"generating hints({len(m_question)})...")
        client = OpenAI(
            base_url = base_url,
            api_key = api_key,
        )
        print("----- standard request -----")
        for idx in range(len(m_question)):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx]
            )
            response = None
            while True:
                try:
                    completion = client.chat.completions.create(
                        model="app-7c54im-1766977238437488331",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who good at math"},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    response = completion.choices[0].message.content
                    break 
                
                except openai.RateLimitError:
                    print(f"Rate limit reached at idx {idx}. Sleeping for 20 seconds...")
                    time.sleep(20)
                except Exception as e:
                    print(f"An unexpected error occurred at idx {idx}: {e}")
                    raise e
                
            hints = extract_hints(response)
            
            # 将当前结果加入列表
            h_question_idx.append(m_question_idx[idx])
            h_question.append(m_question[idx])
            h_hints.append(hints)
            h_ref_solution.append(m_ref_solution[idx])
            h_ref_answer.append(m_ref_answer[idx])

            # -----------------------------------------------------------
            # [修改部分] 每隔 10 条保存一次
            # -----------------------------------------------------------
            if (idx + 1) % 10 == 0:
                print(f"Auto-saving checkpoint at count {idx + 1}...")
                # 注意：m_answer 和 m_entropy 是原始完整列表，
                # 这里使用 [:idx+1] 进行切片，确保传入的长度与当前 h_question 一致
                self.file.save_hints(
                    h_question, 
                    h_hints, 
                    h_ref_solution, 
                    h_ref_answer, 
                    h_question_idx, 
                    m_answer[:idx+1], 
                    m_entropy[:idx+1]
                )
            # -----------------------------------------------------------
            
        print("saving final hints...")
        # 循环结束后保存完整数据（防止总数不是10的倍数导致最后几条没存）
        self.file.save_hints(h_question, h_hints, h_ref_solution, h_ref_answer, h_question_idx, m_answer, m_entropy)
        return True

        



    def teacher_hints_gtp(self) -> bool:
        print("Starting teacher hinting (GPT-4o)...")
        print("load mistakes...")
        self.file.load_mistakes()
        m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution, m_entropy = self.file.parse_data(self.file.mistakes)
        print("mistakes size:", len(m_question))

        h_question = []
        h_hints = []
        h_ref_solution = []
        h_ref_answer = []
        h_question_idx = []

        print(f"generating hints({len(m_question)})...")

        # 初始化 OpenAI 客户端
        # 建议将 key 放入环境变量 OPENAI_API_KEY 中，或者在这里直接替换字符串
        client = OpenAI(
            base_url=base_url,
            api_key = api_key,
            # 如果你使用的是国内中转/代理，取消下面这行的注释并填入地址
            # base_url="https://api.openai-proxy.com/v1" 
        )

        print("----- standard request (GPT-4o) -----")
        for idx in range(len(m_question)):
            prompt = TEACHER_CORRECT_PROMPT.format(
                problem=m_question[idx],
                student_answer=m_answer[idx],
                ref_solution=m_ref_solution[idx]
            )
            
            response = None
            max_retries = 5  # 增加最大重试次数防止死循环
            retry_count = 0

            while retry_count < max_retries:
                try:
                    # 调用 GPT-4o
                    completion = client.chat.completions.create(
                        model="app-7c54im-1766977238437488331", 
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant who is good at math."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7, # 适当增加一点随机性，避免过于死板，或设为0保持确定性
                    )
                    response = completion.choices[0].message.content
                    break 
                
                except RateLimitError:
                    print(f"Rate limit reached at idx {idx}. Sleeping for 20 seconds...")
                    time.sleep(20)
                    retry_count += 1
                except APIError as e:
                    print(f"OpenAI API Error at idx {idx}: {e}. Retrying...")
                    time.sleep(5)
                    retry_count += 1
                except Exception as e:
                    print(f"An unexpected error occurred at idx {idx}: {e}")
                    # 如果是严重错误，可以选择 break 或者 raise
                    raise e
            
            if response:
                hints = extract_hints(response)
                h_question_idx.append(m_question_idx[idx])
                h_question.append(m_question[idx])
                h_hints.append(hints)
                h_ref_solution.append(m_ref_solution[idx])
                h_ref_answer.append(m_ref_answer[idx])
                
                # 打印进度，防止在此处看起来像卡死
                if idx % 5 == 0:
                    print(f"Processed {idx + 1}/{len(m_question)}")
            else:
                print(f"Failed to get response for idx {idx}")

        print("saving hints...")
        self.file.save_hints(h_question, h_hints, h_ref_solution, h_ref_answer, h_question_idx, m_answer, m_entropy)
        return True
       

    def teacher_mark_paper_with_save(self) -> bool:
        incorrect_data, correct_data = self.teacher_mark_paper()
        err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers, err_entropy = incorrect_data
        correct_question_idx, correct_questions, correct_answers, correct_ref_solutions, correct_ref_answers, correct_entropy = correct_data
        self.file.save_mistakes(err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers, err_entropy)
        self.file.save_right(correct_question_idx, correct_questions, correct_answers, correct_ref_solutions, correct_ref_answers, correct_entropy)
        return True
            
    def judge_and_gen_hints(self):
        print("Starting judge and generate hints...")
        self.teacher_mark_paper_with_save()
        self.teacher_hints()
        

    def teacher_mark_paper(self, roll = False):
        print("Starting teacher marking...")
        self.file.load_exam(roll)
        question_idx, question, answer, ref_answer, ref_solution, entropy = self.file.parse_data(self.file.data)
        size = len(question)

        self.acc_count = 0
        self.err_count = 0
        self.toolong_count = 0

        err_question_idx = []
        err_questions = []
        err_answers = []
        err_ref_solutions = []
        err_ref_answers = []
        err_entropy = []
        
        correct_question_idx = []
        correct_questions = []
        correct_answers = []
        correct_ref_solutions = []
        correct_ref_answers = []
        correct_entropy = []
        
        print("----- standard request -----")
        for idx in range(size):
            final_answer = extract_boxed_content(answer[idx])
            final_answer = normalize_answer(final_answer)
            ref_final_answer = normalize_answer(ref_answer[idx])
            
            if final_answer == ref_final_answer:
                self.acc_count += 1
                correct_question_idx.append(question_idx[idx])
                correct_questions.append(question[idx])
                correct_answers.append(answer[idx])
                correct_ref_solutions.append(ref_solution[idx])
                correct_ref_answers.append(ref_answer[idx])
                correct_entropy.append(entropy[idx])
            else:
                self.err_count += 1
                err_question_idx.append(question_idx[idx])
                err_questions.append(question[idx])
                err_answers.append(answer[idx])
                err_ref_solutions.append(ref_solution[idx])
                err_ref_answers.append(ref_answer[idx])
                err_entropy.append(entropy[idx])
            
            if idx % 5 == 0:
                left = size - idx
                print(f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_count}, toolong:{self.toolong_count}")
            
        print(f"Accuracy: {self.acc_count}/{size}")
        print(f"Error count: {self.err_count}")
        
        return (
            (err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers, err_entropy),
            (correct_question_idx, correct_questions, correct_answers, correct_ref_solutions, correct_ref_answers, correct_entropy)
        )



    def check_answers_equivalence(self) -> int:
        print("Loading mistakes for evaluation...")
        self.file.load_mistakes()

        total_questions = len(self.file.mistakes)
        equivalent_count = 0
        
        print(f"Total items to check: {total_questions}")
        print("----- Starting Evaluation (GPT-4o) -----")

        data = self.file.mistakes
        
        # 这里的 client 应该是你在类初始化时创建的
        # 如果没有初始化，请确保 self.client = OpenAI(...) 存在
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://wanqing-api.corp.kuaishou.com/api/agent/v1/apps"
        )
        for idx, item in enumerate(data):
            question_idx = item.get("question_idx", idx)
            question_text = item.get("question", "")
            
            # 获取参考答案
            ref_answer = item.get("ref_answer", "")
            
            # 获取学生答案并尝试提取 boxed 内容 (如果没有 boxed 则用全文)
            # 注意：Prompt 中说 "ignore \boxed", 意味着我们可以传带 boxed 的，也可以传提取后的
            # 这里为了稳妥，传入提取后的核心数值，减少干扰
            raw_answer = item.get("answer", "")
            student_answer_core = extract_boxed_content(raw_answer)

            # 1. 构建 Prompt (注意参数名要和 Prompt 里的占位符一致)
            prompt = OREAL_CORRECT_PROMPT.format(
                question=question_text,
                gold_answer=ref_answer,
                answer=student_answer_core
            )

            is_equivalent = False
            response_content = ""
            
            # 2. 调用 API (包含重试逻辑)
            max_retries = 5
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model="app-7c54im-1766977238437488331", 
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant evaluating math answers."},
                            {"role": "user", "content": prompt},
                        ],
                        # 关键修改：Prompt 要求输出 A/B，不是 JSON，所以去掉 json_object 限制
                        # response_format={"type": "json_object"}, 
                        temperature=0.0, # 判题通常需要确定性，设为 0
                        max_tokens=10    # 我们只需要一个字母，限制 token 节省成本
                    )
                    response_content = completion.choices[0].message.content.strip()
                    break
                
                except RateLimitError:
                    print(f"[Idx {question_idx}] Rate limit reached. Sleeping for 20 seconds...")
                    time.sleep(20)
                    retry_count += 1
                except APIError as e:
                    print(f"[Idx {question_idx}] OpenAI API Error: {e}. Retrying...")
                    time.sleep(5)
                    retry_count += 1
                except Exception as e:
                    print(f"[Idx {question_idx}] Unexpected error: {e}")
                    break
            
            # 3. 解析结果 (文本分类模式)
            # Prompt 要求返回 "A" 或 "B"
            if response_content:
                # 处理可能出现的标点符号，例如 "A."
                clean_resp = response_content.upper().replace(".", "")
                
                if "A" == clean_resp or "CORRECT" in clean_resp:
                    is_equivalent = True
                elif "B" == clean_resp or "INCORRECT" in clean_resp:
                    is_equivalent = False
                else:
                    # 兜底：如果模型话痨了，检查字符串中是否包含明确判定
                    if "CORRECT" in clean_resp and "INCORRECT" not in clean_resp:
                        is_equivalent = True
                    else:
                        print(f"[Idx {question_idx}] Ambiguous response: {response_content}")

            # 4. 统计与记录
            if is_equivalent:
                equivalent_count += 1
                status = "CORRECT"
            else:
                status = "WRONG"

            # 打印简略日志
            print(f"Idx {question_idx}: {status} | GPT Says: {response_content} | Ref: {ref_answer}")

            # 可选：将结果回写到 item 中，方便后续保存
            # item['eval_result'] = status

        # 5. 输出最终统计
        print("-" * 30)
        print(f"Evaluation Finished.")
        print(f"Total Questions: {total_questions}")
        print(f"Equivalent (Correct) Answers: {equivalent_count}")
        if total_questions > 0:
            print(f"Accuracy: {equivalent_count / total_questions * 100:.2f}%")
        
        return equivalent_count

    
if __name__ == "__main__":
    corrector = TeacherCorrecter()
    corrector.check_answers_equivalence()


    