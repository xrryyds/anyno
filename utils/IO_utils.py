import json
from prompt import GEN_ENHANCE_PROMPT
import os

class FileIOUtils:
    def __init__(self):
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file_path)) 
        self.exam_file_path = os.path.join(project_root, "datasets", "exam", "exam.json")
        self.mistake_file_path = os.path.join(project_root, "datasets", "exam", "mistake_collection_book.json")
        self.hints_file_path = os.path.join(project_root, "datasets", "exam", "hints.json")
        self.student_correct_output_path = os.path.join(project_root, "datasets", "exam", "correct.json")




        self.adv_hints_dataset_path = os.path.join(project_root, "datasets", "exam", "adv_hints.json")
        self.disadv_hints_dataset_path = os.path.join(project_root, "datasets", "exam", "disadv_hints.json")
        self.grpo_dataset_path = os.path.join(project_root, "datasets", "exam", "grpo_data.json")
        self.corr_path = os.path.join(project_root, "datasets", "exam", "corr_answer.json")
        self.irdcl_dataset_path = os.path.join(project_root, "datasets", "exam", "irdcl_data.json")
    


        self.data = []
        self.mistakes = []
        self.question_with_hints = []
    
    def load_exam(self) -> bool:
        try:
            with open(self.exam_file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"load fail: {e}")
            return False
        

    def load_mistakes(self) -> bool:
        try:
            with open(self.mistake_file_path, 'r', encoding='utf-8') as f:
                self.mistakes = json.load(f)
            return True
        except Exception as e:
            print(f"load fail: {e}")
            return False
        
    def load_question_with_hints(self) -> bool:
        try:
            with open(self.hints_file_path, 'r', encoding='utf-8') as f:
                self.question_with_hints = json.load(f)
            return True
        except Exception as e:
            print(f"load fail: {e}")
            return False
        
    def parse_data(self, data: list):
        question_idx = []
        question = []
        answer = []
        ref_answer = []
        ref_solution = []
        for idx, item in enumerate(data):
            question_idx.append(item.get("question_idx", "")),
            question.append(item.get("question", ""))
            answer.append(item.get("answer", ""))
            ref_answer.append(item.get("ref_answer", ""))
            ref_solution.append(item.get("ref_solution", ""))
        return question_idx, question, answer, ref_answer, ref_solution
    
    def parse_hints_exam(self, data: list):
        question_idx = []
        question = []
        question_with_hint = []
        hints = []
        ref_answer = []
        ref_solution = []
        student_answer = []
        for idx, item in enumerate(data):
            hint = item.get("hints", "")
            if(hint != ""):
                question.append(item.get("question", ""))
                hints.append(hint)
                ref_answer.append(item.get("ref_answer", ""))
                ref_solution.append(item.get("ref_solution", ""))
                question_idx.append(item.get("question_idx", ""))
                student_answer.append(item.get("student_answer",""))

        for idx in range(len(question)):
            question_with_hint.append(GEN_ENHANCE_PROMPT.format(question=question[idx], hints=hints[idx]))
        return question_idx, question, question_with_hint, ref_solution, ref_answer, student_answer, hints

    def save_hints(self, question: list, hints: list, ref_solution: list, ref_answer: list, question_idx: list,student_answer: list) -> bool:
        try:
            size = len(question)
            data = []
            for idx in range(size):
                item = {
                    "question_idx": question_idx[idx],
                    "question": question[idx],
                    "hints": hints[idx],
                    "ref_solution": ref_solution[idx],
                    "ref_answer": ref_answer[idx],
                    "student_answer": student_answer[idx]
                }
                data.append(item)

            with open(self.hints_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"save fail: {e}")
            return False
        

    def save_mistakes(self, question_idx: list, question: list, answers: list, ref_solution: list, ref_answer: list) -> bool:
        self.save_Q_and_A(question_idx, question, answers, ref_solution, ref_answer, self.mistake_file_path)


    def save_student_correct(self, question_idx: list, question: list, answers: list, ref_solution: list, ref_answer: list) -> bool:
        self.save_Q_and_A(question_idx,question, answers, ref_solution, ref_answer, self.student_correct_output_path)    

    def save_Q_and_A(self, question_idx: list, question: list, answers: list, ref_solution: list, ref_answer: list, path:str) -> bool:
        try:
            size = len(question)
            data = []
            for idx in range(size):
                item = {
                    "question_idx": question_idx[idx],
                    "question": question[idx],
                    "answer": answers[idx],
                    "ref_solution": ref_solution[idx],
                    "ref_answer": ref_answer[idx]
                }
                data.append(item)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"save fail: {e}")
            return False
        
    def save_results_to_json(self, data_list, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved to {path}")
            return True
        except Exception as e:
            print(f"Save fail for {path}: {e}")
            return False

