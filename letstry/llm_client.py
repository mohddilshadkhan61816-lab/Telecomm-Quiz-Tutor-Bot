import os
from typing import Optional

try:
    import openai
except ImportError:
    openai = None

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_openai_api_key() -> Optional[str]:
    return OPENAI_API_KEY


class LLMClient:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key = get_openai_api_key()
        if openai is not None and self.api_key:
            openai.api_key = self.api_key

    def evaluate_answer(self, question: str, user_answer: str, correct_answer: str, explanation: str, context: str) -> str:
        if not self.api_key or openai is None:
            return self._fallback_evaluation(question, user_answer, correct_answer, explanation, context)

        prompt = (
            "You are a telecom training assistant. A learner answered a quiz question. "
            "Provide an evaluation that includes whether the answer is correct, a score from 0 to 100, "
            "and a detailed explanation. If the answer is wrong, compare the answer to the correct concept and use the context information." 
            f"\n\nQuestion: {question}\n"
            f"Learner Answer: {user_answer}\n"
            f"Correct Answer: {correct_answer}\n"
            f"Context: {context}\n"
            f"Key Explanation: {explanation}\n"
        )

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are a helpful telecom education assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _fallback_evaluation(question: str, user_answer: str, correct_answer: str, explanation: str, context: str) -> str:
        if correct_answer.lower() in user_answer.lower():
            score = 90
            detail = (
                "Your answer aligns well with the expected telecom concept. "
                "Review the context for additional details to deepen your understanding."
            )
        else:
            score = 45
            detail = (
                "The answer does not match the expected plan detail. "
                "Study the plan context and focus on the specific service attributes mentioned."
            )

        return (
            f"Evaluation:\n"
            f"Score: {score}/100\n"
            f"Correct Answer: {correct_answer}\n"
            f"Feedback: {detail}\n"
            f"Explanation: {explanation}\n"
            f"Context: {context}"
        )
