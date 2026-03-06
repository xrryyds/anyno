QUESTION_PROMPT  = """
# Question:
Please solve the following math problem.
## Strictly adhere to the output format below:

1. **Reasoning**: Wrap your step-by-step deduction process inside <thinking>...</thinking> tags. This section must not exceed {max_token} words.
2. **Result**: Wrap the final numerical answer inside <answer>...</answer> tags. Do not include units or text in this tag.


Problem:
{problem_text}

## Your output must exactly match this structure:
Thinking:
<thinking>
[Your derivation steps here]
</thinking>

Answer:
<answer>
[Final number here]
</answer>

## Please answer below:
"""

ANSWER_PROMPT = """
Thinking:
<thinking>
{thinking}
</thinking>

Answer:
<answer>
{answer}
</answer>
"""

JUDGE_PROMPT = """
{question_and_answer}

# Judge:
You are an expert mathematics grader. You will be provided with a **Math Problem** and a **Model Response** (consisting of a `<thinking>` process and an `<answer>`).
Your goal is to verify whether the Model Response is mathematically correct.

# Input Data:
Problem:
{problem_text}

Model Response:
{model_response}

# Evaluation Criteria:
1. **Accuracy**: Is the final numerical value in the `<answer>` tag correct?
2. **Logic**: Is the reasoning process in the `<thinking>` tag mathematically sound and consistent with the answer? This section must not exceed {max_token} words.

# Output Format:
Please strictly adhere to the following XML format for your evaluation:

<conclusion>[YES or NO]</conclusion>
<reason>[Your explanation]</reason>

# Rules for Filling Tags:
1. **<conclusion>**:
   - Return **YES** if the solution is entirely correct.
   - Return **NO** if there is any error in the logic or the final answer.

2. **<reason>**:
   - If the conclusion is **YES**, simply write: "Correct".
   - If the conclusion is **NO**, concisely explain the specific reason for rejection (e.g., "Calculation error in step 2", "Final answer is wrong", or "Logic does not follow the problem conditions").
"""

JUDGER_GEN_REASON_PROMPT= """

<reason>{reason}</reason>
"""


#############################################################################################
# Mode A: 输入包含 Hint
GEN_ENHANCE_PROMPT = """{question}{hints}
"""

# Mode B: 输入只包含 Question
GEN_PROMPT = """{question}
"""

# Mode B: 模型的输出目标 (Hint + Answer)
# 注意：这里开头补上了 prompt 结尾可能需要的换行或标识
GEN_HINTS_WIH_ANSWER = """{hints}{answer}"""



TEACHER_CORRECT_PROMPT = """
**Role:** Heuristic Logic Mentor & Knowledge Bridge

**Task:**
Your task is to analyze the **Student's Solution** in comparison to the **Reference Answer**. Instead of acting as a simple grader, you must identify the specific **missing logical link** or **knowledge gap** that prevents the student from reaching the correct conclusion.

**Goal:**
Provide the student with the necessary "Knowledge Hints" so that, based on their existing correct reasoning, they can bridge the gap and solve the problem themselves.

**Critical Constraints for Hints:**
1.  **Universal Knowledge Only:** Hints must be provided as **general problem-solving methods, formulas, theorems, or definitions** (e.g., "Recall the formula for the area of a circle: $S = \pi r^2$" or "Use the derivatives of trigonometric functions: $(sin x)' = cos x$").
2.  **No Specific Calculations:** Do not calculate the result for the specific numbers in the problem. Provide the *tool*, not the *answer*.
3.  **Targeted:** The hint must directly address the specific step where the student's logic broke down or stopped.
4.  **Token Limit:** Each individual hint must be strictly limited to **50 tokens maximum**. Keep descriptions concise and focus purely on the core principle.
5.  **Unidentifiable Gap:** If it is impossible to determine if the student lacks specific universal knowledge (e.g., the error is purely a calculation/arithmetic mistake, a typo, the answer is empty/irrelevant, or the reasoning is too unclear to pinpoint a missing theorem), return `null`.

# Problem:
{problem}

# Student's Answer:
{student_answer}

# Reference Answer:
{ref_solution}

**Output Format:**

**Condition A:** If Constraint #5 applies (unable to identify a specific general knowledge gap), output strictly:
`null`

**Condition B:** If a knowledge gap is identified, respond in the following XML format:
<hints>
[Provide the general formula, theorem, or principle. Use LaTeX for math expressions.]
...
</hints>

**Example:**
*   **Student's Error:** Calculated probability as $P(A)+P(B)$ but events were not mutually exclusive.
*   **Your Output:**
<hints>
For any two events $A$ and $B$, the probability of their union is $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
</hints>
"""



OREAL_CORRECT_PROMPT = """You are a helpful assistant who evaluates the correctness and quality of models' outputs.
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: CORRECT
    B: INCORRECT
    Just return the letters \"A\" or \"B\", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>:
    {question}
    <Original Question End>


    <Gold Target Begin>:
    {gold_answer}
    <Gold Target End>


    <Predicted Answer Begin>:
    {answer}
    <Predicted End>

    Judging the correctness of candidates' answers:"""


