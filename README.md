# Enterprise Document Q&A

![CI](https://github.com/Mohsen-Fekri/enterprise-document-qa/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
[![Live demo](https://img.shields.io/badge/Live%20demo-try%20it%20in%20your%20browser-1D4ED8)](https://mohsen-fekri.github.io/enterprise-document-qa/)

A question-answering assistant for company documents like security policies, audit procedures and risk reports. It reads PDF and Word files, finds the passages that answer your question, and replies with page-level citations. If the documents it's allowed to search don't contain enough evidence, it says so instead of making something up.

Each user also has a role, and the role decides which documents they can search. That check happens during retrieval, so restricted files never get near the answer in the first place.

The backend is FastAPI and the UI is Streamlit. You don't need an API key to run it: without one, answers are quoted straight from the retrieved passages. With `OPENAI_API_KEY` set, it writes generated answers and can run RAGAS scoring.

**Live demo:** https://mohsen-fekri.github.io/enterprise-document-qa/

Pick a role and ask a question. The demo is a JavaScript port of the same retrieval and answer code, running on the sample documents in your browser, so it needs no server or API key.

![A compliance officer's question answered from the documents, with page citations and a confidence score.](docs/grounded-answer.png)

## What it does

- Reads PDF and Word (.docx) files
- Splits them into chunks of 256, 512 or 1024 characters
- Searches with BM25, TF-IDF, or a hybrid of both (reciprocal rank fusion)
- Cites the page for every answer
- Refuses to answer when the retrieved evidence is weak
- Role-based access for admin, compliance, auditor and employee users
- Local evaluation metrics, plus optional RAGAS

## Roles

```
Streamlit UI  --HTTP-->  FastAPI
                           |
              ingest -> chunk -> retrieve -> answer
                           |
                    evaluation (local / RAGAS)
```

| Role | Can search | Can upload | Can run evaluation |
| --- | --- | --- | --- |
| Administrator | Everything | Yes | Yes |
| Compliance officer | Public up to Restricted | Yes | Yes |
| Internal auditor | Public up to Confidential | No | Yes |
| Employee | Public and Internal only | No | No |

Demo logins: `admin/admin123`, `compliance/comp123`, `auditor/audit123`, `employee/emp123`. They're only for local testing, so change them before deploying anywhere.

### Access control happens at retrieval

Here's the same kind of question from two different users:

![An employee asks about the incident response plan and gets no answer, because that document is Restricted and is filtered out before retrieval.](docs/rbac-abstention.png)

Jordan Hale (compliance officer) asks "What is the SLA for remediating critical audit findings?" and gets an answer citing the Internal Audit Procedure, page 2, at 80% confidence. Riley Patel (employee) asks about P1 incident response and the assistant declines at 33% confidence. The incident response plan is Restricted, so it was never in Riley's search results. The Evaluate tab is hidden for Riley for the same reason.

The point is that the assistant can't answer because it can't see the evidence, not because something hid the answer after the fact.

## Running it

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PYTHONPATH=.
python -m scripts.generate_sample_docs
python -m scripts.seed_corpus

# terminal 1: API
uvicorn backend.main:app --host 0.0.0.0 --port 8765

# terminal 2: UI
export BACKEND_URL=http://127.0.0.1:8765
streamlit run frontend/app.py --server.port 8502
```

Or start both with `bash scripts/dev.sh` and open http://127.0.0.1:8502. The API docs are at http://127.0.0.1:8765/docs.

## Questions to try

Signed in as admin or compliance, these should get answers:

- How often must employees rotate their passwords?
- What is the SLA for remediating critical audit findings?
- What is the residual risk rating for ransomware?
- What is the P1 incident response time?

These should get a polite "not in the documents":

- What is the CEO salary for 2026?
- Where is the Tokyo branch office located?

Signed in as an employee, the P1 question should also be declined, because the incident response plan is Restricted.

## Evaluation

Run it from the Evaluate tab, or:

```bash
export PYTHONPATH=.
python -m scripts.evaluate_local
```

Without an LLM I measure: word overlap between the answer and the retrieved context, word overlap with a reference answer, context recall on a small gold set, whether it declined when it should have, and citation coverage. These are rough lexical checks, so I don't call them faithfulness or relevance scores. With `OPENAI_API_KEY` set (copy `.env.example` to `.env`), RAGAS adds faithfulness, answer relevancy, context precision and context recall.

I tried all nine combinations of the three chunk sizes and three retrievers. The results are in `data/eval/local_results.json`. The best one on my 11-question test set:

| Setting | Result |
| --- | ---: |
| Chunk size | 512 characters |
| Retrieval | Hybrid BM25 + TF-IDF |
| Answer-context overlap | 0.967 |
| Answer-reference overlap | 0.293 |
| Context recall | 1.000 |
| Correct refusals | 1.000 |
| Citation coverage | 1.000 |
| Overall local score | 0.771 |

## Limitations

- The documents are generated for this demo. There are no real company files in the repo.
- 11 questions is enough to catch regressions, not to benchmark the system properly.
- The local metrics compare words, not meaning, so they can't tell whether an answer is actually correct.
- Word files don't store page numbers, so I estimate pages from paragraph length.
- The demo passwords and the default JWT secret have to be replaced before any real deployment.
- Storage is a simple local store meant for a demo, not for many users at once.

## Tests

```bash
export PYTHONPATH=.
pytest -q
```

## Project layout

```
backend/main.py          FastAPI routes
frontend/app.py          Streamlit UI
enterprise_qa/           retrieval, access control, ingest, evaluation
scripts/                 sample documents, seeding, dev runner
docs/                    browser demo (GitHub Pages) and screenshots
data/sample_docs/        generated PDF and Word files
data/eval/gold_qa.json   test questions
tests/
```

## Author

Mohsen Fekri, Computer Science, York University · [LinkedIn](https://www.linkedin.com/in/mohsen-fekri) · justinfe@my.yorku.ca

## License

MIT
