from utils import FileIOUtils, extract_hints ,extract_boxed_content, normalize_answer
from openai import OpenAI
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
        m_question_idx, m_question, m_answer, m_ref_answer, m_ref_solution = self.file.parse_data(self.file.mistakes)
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
            h_question_idx.append(m_question_idx[idx])
            h_question.append(m_question[idx])
            h_hints.append(hints)
            h_ref_solution.append(m_ref_solution[idx])
            h_ref_answer.append(m_ref_answer[idx])
            
        print("saving hints...")
        self.file.save_hints(h_question, h_hints, h_ref_solution, h_ref_answer, h_question_idx, m_answer)
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
        

    def teacher_mark_paper(self):
        print("Starting teacher marking...")
        self.file.load_exam()
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
                err_ref_answers.append(entropy[idx])
            
            if idx % 5 == 0:
                left = size - idx
                print(f"finished: {idx}, left: {left}, acc:{self.acc_count}, err:{self.err_count}, toolong:{self.toolong_count}")
            
        print(f"Accuracy: {self.acc_count}/{size}")
        print(f"Error count: {self.err_count}")
        
        return (
            (err_question_idx, err_questions, err_answers, err_ref_solutions, err_ref_answers, err_entropy),
            (correct_question_idx, correct_questions, correct_answers, correct_ref_solutions, correct_ref_answers, correct_entropy)
        )


    
if __name__ == "__main__":
    corrector = TeacherCorrecter()
    # corrector.judge_and_gen_hints()
    corrector.teacher_mark_paper()
    


    