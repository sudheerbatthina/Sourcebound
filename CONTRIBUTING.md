# Contributing to Sourcebound

Thanks for helping improve Sourcebound. This project welcomes focused fixes, documentation improvements, tests, and feature proposals that make private document Q&A more useful and reliable.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/<your-org>/sourcebound.git
cd sourcebound
```

Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Code Style

Ruff is enforced for linting. Run it before opening a pull request:

```bash
ruff check .
```

Keep changes focused and follow the existing project style.

## Running Tests

Run the smoke test suite:

```bash
python scripts/ci_test.py
```

Some evaluation flows may require API keys. The CI smoke tests are designed to run without secrets.

## PR Process

1. Fork the repository.
2. Create a feature branch from the default branch.
3. Make your changes with tests or documentation updates when relevant.
4. Open a pull request.
5. Participate in review and address requested changes.

## Development Setup

Copy the example environment file and fill in local values:

```bash
cp .env.example .env
```

At minimum, set `OPENAI_API_KEY` for embedding and answer generation flows. Set `API_KEYS` to the comma-separated API keys accepted by the FastAPI service.

Run the API locally:

```bash
uvicorn api:app --reload
```

Run the Streamlit UI locally:

```bash
streamlit run app.py
```
