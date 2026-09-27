# AI Engineer Interview Questions

566 AI engineering interview questions across 15 topics, each answered so that someone new to the
topic can follow it: the answer in plain English first, then the idea with a small example, how it works step by
step with every term defined, the formula read out symbol by symbol where there is one, and a figure to look at.

## Topics

| # | Topic | Questions |
|---|---|---|
| 1 | [Must Know](topics/00-must-know.md) | 6 |
| 2 | [LLM Fundamentals](topics/01-llm-fundamentals.md) | 78 |
| 3 | [Prompt Engineering](topics/02-prompt-engineering.md) | 30 |
| 4 | [Retrieval-Augmented Generation (RAG)](topics/03-retrieval-augmented-generation-rag.md) | 41 |
| 5 | [AI Agents and Agentic Systems](topics/04-ai-agents-and-agentic-systems.md) | 53 |
| 6 | [Fine-Tuning and Model Adaptation](topics/05-fine-tuning-and-model-adaptation.md) | 32 |
| 7 | [Vector Databases and Embeddings](topics/06-vector-databases-and-embeddings.md) | 26 |
| 8 | [AI System Design](topics/07-ai-system-design.md) | 49 |
| 9 | [LLMOps and Production AI](topics/08-llmops-and-production-ai.md) | 52 |
| 10 | [Evaluation and Testing](topics/09-evaluation-and-testing.md) | 36 |
| 11 | [AI Safety, Ethics, and Responsible AI](topics/10-ai-safety-ethics-and-responsible-ai.md) | 45 |
| 12 | [Multimodal AI](topics/11-multimodal-ai.md) | 30 |
| 13 | [AI Infrastructure and Scalability](topics/12-ai-infrastructure-and-scalability.md) | 34 |
| 14 | [Coding and Practical Implementation](topics/13-coding-and-practical-implementation.md) | 32 |
| 15 | [Behavioral and Scenario-Based Questions](topics/14-behavioral-and-scenario-based-questions.md) | 22 |

## How to use it

- Read the plain-English answer, then try to explain the idea out loud before reading how it works.
- Follow the figure alongside the text; each answer says what to look at in it.
- Anything that changes fast (model names, versions, prices, regulations) is dated; check it before you quote it.
- Numbers are labelled as a typical range, a published result or a rule of thumb; scenario numbers are illustrative.

## How it is built

Figures are hand-built SVGs (one script each in `diagrams/`, drawn with `scripts/svg.py`), charts are Vega-Lite
specs in `charts/` rendered by `scripts/vl2svg.mjs`, and formulas are LaTeX. `scripts/check.py` and
`scripts/check-math.mjs` check the questions, links, lengths and every formula.
