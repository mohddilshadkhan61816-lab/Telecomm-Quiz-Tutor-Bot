# Telecomm Quiz/Tutor Bot

An interactive AI-driven educational bot for the telecommunication domain. The bot uses plan details and targeted assessment questions to evaluate learner responses, score answers, and provide contextual explanations for incorrect responses.

## Features

- Vectorized knowledge retrieval from telecom plan details
- Quiz generation with targeted telecom questions
- Answer evaluation using a Large Language Model (OpenAI)
- Detailed explanations for incorrect answers
- Personalized learning loop with industry-specific feedback

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key in an environment variable:

```bash
set OPENAI_API_KEY=your_api_key_here
```

3. Run the browser-based tutor:

```bash
python app.py
```

4. Open your browser and visit:

```bash
http://127.0.0.1:5000
```

## Data

`data/plan_details.json` contains telecom plan content and quiz questions used to build the vector database.

## Notes

- If `OPENAI_API_KEY` is not set, the app will still run but evaluation and explanation will use a local fallback logic.
- The vector database is in-memory and built from the plan knowledge base when the app starts.
