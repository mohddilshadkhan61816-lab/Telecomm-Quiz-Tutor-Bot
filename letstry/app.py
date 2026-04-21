import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from llm_client import LLMClient
from quiz_engine import build_context_text, build_vector_db, load_knowledge_base, retrieve_context

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")

knowledge_base = load_knowledge_base()
plan_contents = knowledge_base.get("plan_details", [])
quiz_questions = knowledge_base.get("quiz_questions", [])
vector_store = build_vector_db(plan_contents)
llm_client = LLMClient()


def find_question(question_id: str) -> Dict[str, Any]:
    for question in quiz_questions:
        if question.get("id") == question_id:
            return question
    return {}


@app.route("/")
def index() -> str:
    return render_template(
        "index.html",
        questions=json.dumps(quiz_questions),
    )


@app.route("/evaluate", methods=["POST"])
def evaluate() -> Any:
    data = request.get_json(force=True)
    question_id = data.get("question_id", "")
    user_answer = data.get("user_answer", "").strip()

    question = find_question(question_id)
    if not question:
        return jsonify({"error": "Question not found."}), 400

    context_docs = retrieve_context(vector_store, user_answer)
    context_text = build_context_text(context_docs)
    evaluation = llm_client.evaluate_answer(
        question=question["question"],
        user_answer=user_answer,
        correct_answer=question["answer"],
        explanation=question["explanation"],
        context=context_text,
    )

    correct = question["answer"].lower() in user_answer.lower()
    snippet_cards = [
        {"title": doc["title"], "content": doc["content"]}
        for doc in context_docs
    ]

    return jsonify(
        {
            "question_id": question_id,
            "correct": correct,
            "evaluation": evaluation,
            "correct_answer": question["answer"],
            "snippet_cards": snippet_cards,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=True)
