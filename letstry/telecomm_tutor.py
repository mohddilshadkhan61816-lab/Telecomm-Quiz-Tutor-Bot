import os
import random
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from llm_client import LLMClient
from quiz_engine import DATA_FILE, build_context_text, build_vector_db, load_knowledge_base, retrieve_context

load_dotenv()


def ask_user_answer(question_data: Dict[str, Any]) -> str:
    print("\nQuestion:")
    print(question_data["question"])
    for option_index, option in enumerate(question_data["options"], start=1):
        print(f"  {option_index}. {option}")

    while True:
        answer_text = input("Enter your answer (text or option number): ").strip()
        if not answer_text:
            print("Please enter a valid answer.")
            continue

        if answer_text.isdigit():
            choice_index = int(answer_text) - 1
            if 0 <= choice_index < len(question_data["options"]):
                return question_data["options"][choice_index]
            print("Option number is out of range. Please try again.")
            continue

        return answer_text


def run_quiz(knowledge_base: Dict[str, Any], llm_client: LLMClient) -> None:
    questions = knowledge_base.get("quiz_questions", [])
    plan_contents = knowledge_base.get("plan_details", [])
    random.shuffle(questions)

    store = build_vector_db(plan_contents)
    total_score = 0
    points_per_question = 100 // max(1, len(questions))

    print("\nWelcome to the Telecomm Quiz/Tutor Bot!\nProvide your answer and get a tailored explanation for any incorrect response.")

    for question_data in questions:
        user_answer = ask_user_answer(question_data)
        context_docs = retrieve_context(store, user_answer)
        context_text = build_context_text(context_docs)

        evaluation = llm_client.evaluate_answer(
            question=question_data["question"],
            user_answer=user_answer,
            correct_answer=question_data["answer"],
            explanation=question_data["explanation"],
            context=context_text,
        )

        print("\n--- Evaluation Result ---")
        print(evaluation)

        if question_data["answer"].lower() in user_answer.lower():
            total_score += points_per_question

        if question_data["answer"].lower() not in user_answer.lower():
            print("\nRelevant knowledge snippets used for explanation:")
            for idx, doc in enumerate(context_docs, start=1):
                print(f"\n[{idx}] {doc['title']}\n{doc['content']}")

    print(f"\nFinal estimated score: {total_score}/{points_per_question * len(questions)}")
    print("Thank you for using the Telecomm Quiz/Tutor Bot. Review the explanations to strengthen your telecom domain knowledge.")


def main() -> None:
    if not os.path.exists(DATA_FILE):
        print(f"Knowledge base file not found: {DATA_FILE}")
        sys.exit(1)

    knowledge_base = load_knowledge_base(DATA_FILE)
    llm_client = LLMClient()
    if llm_client.api_key is None:
        print("Warning: OPENAI_API_KEY not found. Running in fallback explanation mode.")

    run_quiz(knowledge_base, llm_client)


if __name__ == "__main__":
    main()
