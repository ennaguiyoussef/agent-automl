# Agentic AutoML

Upload a CSV, describe your goal in plain language, and watch an autonomous agent
**plan, profile, preprocess, train, evaluate, and explain** a set of machine-learning
models — then download the best one. Every number you see is computed in Python;
the LLM only *reasons about and interprets* those numbers. It never invents a metric
and never executes arbitrary code.

> **Core principle — Python calculates, the LLM interprets.**
> Deterministic ML tools do all the math and model fitting. The LLM is confined to
> planning and explanation, and is always forced to return **structured, validated
> output** (Pydantic). With no API key configured, the whole system still runs end-to-end
> in a deterministic **heuristic mode** — it degrades, it doesn't fake.

---

## Table of contents

1. [Overview](#1-overview)
2. [What makes it *agentic*](#2-what-makes-it-agentic)
3. [Architecture](#3-architecture)
4. [The LangGraph workflow](#4-the-langgraph-workflow)
5. [Tech stack](#5-tech-stack)
6. [Quick start (Docker)](#6-quick-start-docker)
7. [Local development](#7-local-development)
8. [Configuration](#8-configuration)
9. [API reference](#9-api-reference)
10. [Project structure](#10-project-structure)

---

## 1. Overview

Agentic AutoML is a full-stack demonstration of a **real agentic workflow** — not a thin
wrapper around `sklearn.pipeline`. A [LangGraph](https://langchain-ai.github.io/langgraph/)
state machine drives a sequence of specialized agents. Each agent has one job; some call an
LLM to *reason*, all of them call deterministic Python **tools** to *act*. The result of every
step is written into a single typed `AutoMLState`, streamed to the browser over Server-Sent
Events so you can watch the pipeline think in real time.

The user experience:

1. **Upload** a `.csv`. The backend profiles it (shape, dtypes, missingness, cardinality,
   duplicates, constant columns, class balance).
2. **State a goal** ("predict which customers churn"). Optionally pin a target column or a
   metric — or let the agent decide.
3. **Watch the pipeline run** station by station on a live activity rail.
4. **Confirm the target** if the agent isn't confident (human-in-the-loop pause).
5. **Review results**: a leaderboard chart, the best-model card with cross-validated scores,
   and an LLM-written review grounded strictly in the computed metrics.
6. **Download** the trained model (`.pkl`) and a JSON report.

---

## 2. What makes it *agentic*

This project deliberately satisfies a set of constraints that separate an agent from a script:

| Constraint | How it's honored |
|---|---|
| **Real agentic workflow** | A LangGraph `StateGraph` with a shared typed state, conditional routing, a checkpointer, and an `interrupt()`-based human-in-the-loop step. |
| **LLM reasoning separated from ML execution** | LLM agents (planner, reviewer, target confirmation) only *choose* and *explain*. All computation lives in `app/ml/*` and `app/tools/*`. |
| **No arbitrary code execution** | The LLM selects from a fixed registry of models and preprocessing strategies. It cannot emit or run code. |
| **Structured output** | Every LLM call returns a Pydantic model via `structured_invoke(...)`; malformed responses are rejected and retried. |
| **No hard-coded secrets** | Provider + key + model are read from environment variables through `app/config.py`. |
| **No invented metrics** | The reviewer is handed only the metrics dict produced by Python and is instructed to interpret, never to compute. |
| **Graceful degradation** | If no LLM key is present (or a call fails), deterministic heuristics take over — the app never fabricates results. |

---

## 3. Architecture

```
┌──────────────────────────┐         SSE (live agent events)        ┌───────────────────────────┐
│   React + TypeScript UI   │  ◀───────────────────────────────────  │        FastAPI API        │
│  (Vite, Tailwind, shadcn) │                                         │  thin routes, no logic    │
│                           │  ── upload / create / chat / results ─▶ │                           │
└──────────────────────────┘                                         └─────────────┬─────────────┘
                                                                                    │
                                                          ┌─────────────────────────▼─────────────────────────┐
                                                          │            Experiment Manager (service)            │
                                                          │  runs the graph in a worker thread, streams events │
                                                          └─────────────────────────┬─────────────────────────┘
                                                                                    │
                                              ┌─────────────────────────────────────▼─────────────────────────────────────┐
                                              │                         LangGraph  StateGraph                              │
                                              │  planner → dataset_analyzer → eda → target_confirmation → preprocessing →  │
                                              │  model_selection → training → evaluation → model_reviewer → artifacts      │
                                              └──────────────┬──────────────────────────────────────┬────────────────────┘
                                                             │ reasons                               │ acts
                                                   ┌─────────▼─────────┐                   ┌──────────▼──────────┐
                                                   │   LLM (LangChain) │                   │  Deterministic ML   │
                                                   │  structured only  │                   │  tools + sklearn    │
                                                   └───────────────────┘                   └─────────────────────┘
```

**Separation of concerns**
- **`app/api`** — HTTP surface only. Routes validate input, call a service, translate domain
  exceptions to status codes. No business logic.
- **`app/services`** — orchestration: experiment lifecycle, the event bus (bridges worker-thread
  graph nodes to the async SSE stream), dataset storage, CSV validation.
- **`app/graph`** — the LangGraph `StateGraph` definition and the typed `AutoMLState`.
- **`app/agents`** — one module per node. LLM helpers (`llm.py`) and deterministic fallbacks
  (`heuristics.py`) sit side by side.
- **`app/ml`** — the model registry, preprocessing, training/CV, and metric computation. **All math lives here.**
- **`app/tools`** — dataset & EDA profiling functions the agents call.
- **`app/schemas`** — Pydantic models shared by the API and the agents.

---

## 4. The LangGraph workflow

The graph is a linear pipeline over a shared, typed `AutoMLState`, with an `interrupt()`-based
human-in-the-loop pause inside the `target_confirmation` node. Each node emits
`*_started` / `*_completed` events that stream to the UI.

```
START
  └─▶ planner              interpret the goal → structured MLPlan (target guess, problem type, metric)
  └─▶ dataset_analyzer     profile columns, dtypes, missingness, cardinality, leakage risks
  └─▶ eda                  summary statistics, correlations, target signal
  └─▶ target_confirmation  confident? continue :: unsure? interrupt() and ask the user  ◀── human-in-the-loop
  └─▶ preprocessing        choose imputers / encoders / scalers (no leakage: fit on train only)
  └─▶ model_selection      pick candidate models from the registry for the problem type
  └─▶ training             fit each candidate, cross-validate, time it
  └─▶ evaluation           score on held-out test set, rank, build the leaderboard
  └─▶ model_reviewer       LLM interprets the computed metrics → recommendation + caveats
  └─▶ artifact_generation  persist best model (.pkl) + JSON report
END
```

- **Human-in-the-loop.** When the planner/analyzer can't confidently identify the target,
  `target_confirmation` calls LangGraph's `interrupt()`. The API surfaces a
  `pending_human` request; the UI shows a confirmation banner; the user's answer resumes the
  graph via `Command(resume=...)`.
- **Checkpointing.** A `MemorySaver` checkpointer makes the interrupt/resume cycle possible and
  keeps state consistent across the worker thread and the API.
- **Leakage-aware.** Preprocessing transformers are fit on the training split only; the target is
  excluded from features; constant and high-cardinality columns are handled explicitly.

---

## 5. Tech stack

**Backend**
- Python 3.11+ · **FastAPI** · **uvicorn** · **sse-starlette** (Server-Sent Events)
- **LangGraph** (agent orchestration) · **LangChain** + **langchain-groq** (model abstraction)
- **Pydantic** / **pydantic-settings** (typed schemas + config)
- **scikit-learn**, **pandas**, **numpy**, **joblib** (all ML computation)
- **pytest** (25 tests: tools, preprocessing, training, EDA, graph smoke, API)
- **uv** for dependency management

**Frontend**
- **React 18** + **TypeScript** + **Vite 5**
- **Tailwind CSS 3.4** + hand-written **shadcn/ui**-style primitives (cva + clsx + tailwind-merge)
- **lucide-react** icons · **Recharts** (leaderboard chart) · dark/light theming via CSS variables

**Ops**
- **Docker** multi-stage builds · **docker-compose** · **nginx** (serves the SPA and reverse-proxies the API)

---

## 6. Quick start (Docker)

Prerequisites: Docker + Docker Compose.

```bash
# 1. Configure environment (copy the template and add a key if you have one)
cp .env.example .env
#    edit .env → set GROQ_API_KEY=...  (optional: omit to run in heuristic mode)

# 2. Build and start both services
docker compose up --build
```

Then open **http://localhost:5173**.

- The frontend is served by nginx and proxies `/api` (including the SSE stream) to the backend,
  so everything is same-origin — no CORS to configure.
- Uploaded datasets and trained artifacts are persisted in named volumes (`backend-data`,
  `backend-artifacts`), so they survive restarts.
- The backend exposes its own port at **http://localhost:8000** (interactive docs at
  `http://localhost:8000/docs`).

To stop: `docker compose down` (add `-v` to also drop the data/artifact volumes).

---

## 7. Local development

Run the two services directly for hot-reload during development.

**Backend** (from `backend/`, using [uv](https://docs.astral.sh/uv/)):

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Run the tests:

```bash
cd backend
uv run pytest -q
```

**Frontend** (from `frontend/`):

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on **http://localhost:5173** and proxies `/api` to
`http://localhost:8000` (configurable via `VITE_API_BASE_URL`). Production build + type-check:

```bash
npm run build      # tsc --noEmit && vite build
```

---

## 8. Configuration

All configuration is environment-driven (see [`.env.example`](.env.example) for the full,
commented list). Never commit real keys.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `openai`. |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model id for the provider. (`llama-3.3-70b-versatile` is decommissioned on Groq.) |
| `GROQ_API_KEY` / `OPENAI_API_KEY` | — | Provider key. **If unset, the app runs in deterministic heuristic mode.** |
| `LLM_API_KEY` | — | Generic override that takes precedence over provider-specific keys. |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature for reasoning calls. |
| `LLM_TIMEOUT` | `60` | Per-call timeout (seconds). |
| `CORS_ORIGINS` | localhost:5173/3000 | Allowed origins (comma-separated or JSON array). |
| `MAX_UPLOAD_MB` | `50` | Upload size cap. |
| `MAX_ROWS` / `MAX_COLUMNS` | `200000` / `2000` | Dataset guardrails. |
| `RANDOM_STATE` | `42` | Global seed for reproducible experiments. |
| `TEST_SIZE` | `0.2` | Held-out test fraction. |
| `CV_FOLDS` | `5` | Cross-validation folds. |
| `MAX_TRAINING_ROWS` | `50000` | Row cap for training (sampled if exceeded). |
| `DATA_DIR` / `ARTIFACTS_DIR` | inside `backend/` | Storage locations (set to volume mounts in Docker). |
| `VITE_API_BASE_URL` | `http://localhost:8000` | API base for the frontend (empty ⇒ relative `/api`). |

---

## 9. API reference

Base URL: `/api`. Interactive OpenAPI docs at `/docs`.

| Method | Path | Description |
|---|---|---|
| `GET`  | `/api/health` | Liveness + whether an LLM is available and which provider. |
| `POST` | `/api/datasets/upload` | Multipart CSV upload → validated `DatasetInfo` profile. |
| `POST` | `/api/experiments` | Create an experiment (`dataset_id`, `user_query`, optional `target_column`, `optimize_metric`). Starts the graph. |
| `GET`  | `/api/experiments` | List experiment snapshots. |
| `GET`  | `/api/experiments/{id}` | Authoritative snapshot: status, plan, target/problem/metric, `pending_human`, `has_results`. |
| `GET`  | `/api/experiments/{id}/events` | **SSE** stream of named agent events (with full backlog replay on connect). |
| `POST` | `/api/experiments/{id}/chat` | Ask the agent a grounded question about this run. |
| `POST` | `/api/experiments/{id}/human-response` | Resume a paused run by confirming the target column. |
| `GET`  | `/api/experiments/{id}/results` | Final evaluation, leaderboard, best-model summary, and LLM review. |
| `GET`  | `/api/experiments/{id}/download-model` | Download the best trained model (`.pkl`). |
| `GET`  | `/api/experiments/{id}/download-report` | Download the JSON run report. |

---

## 10. Project structure

```
.
├── docker-compose.yml          # full-stack orchestration
├── .env.example                # commented configuration template
├── README.md
│
├── backend/
│   ├── Dockerfile              # uv-based image → uvicorn
│   ├── pyproject.toml          # backend deps (FastAPI, LangGraph, sklearn, …)
│   └── app/
│       ├── main.py             # FastAPI app: CORS, event bus, exception→HTTP mapping
│       ├── config.py           # env-driven settings (no hard-coded secrets)
│       ├── api/                # thin HTTP routes (datasets, experiments)
│       ├── services/           # experiment manager, event bus, storage, CSV validation
│       ├── graph/              # LangGraph StateGraph + typed AutoMLState
│       ├── agents/             # one module per node; llm.py + heuristics.py
│       ├── ml/                 # model registry, preprocessing, training/CV, metrics
│       ├── tools/              # dataset + EDA profiling tools
│       ├── schemas/            # shared Pydantic models
│       └── tests/              # pytest suite (25 tests)
│
└── frontend/
    ├── Dockerfile              # multi-stage: Vite build → nginx
    ├── nginx.conf              # SPA fallback + /api reverse proxy (SSE-friendly)
    ├── vite.config.ts          # dev proxy + @ alias
    └── src/
        ├── App.tsx             # shell: sidebar, upload→compose→experiment flow, chat rail
        ├── lib/                # api client, types, pipeline-stage derivation, formatters
        ├── hooks/              # useExperiment (poll), useEventStream (SSE), useTheme, chart colors
        └── components/
            ├── layout/         # sidebar, logo, theme toggle
            ├── dataset/        # upload hero, goal composer, dataset profile
            ├── pipeline/       # PipelineRail (live station view), EventLog
            ├── hitl/           # target-confirmation banner
            ├── chat/           # grounded chat panel
            ├── results/        # leaderboard, best-model card, review, results panel
            └── workspace/      # ExperimentWorkspace (assembles the experiment view)
```

---

### Design notes

- **The reviewer cannot lie about numbers.** It receives the computed metrics dict and is prompted
  to interpret only. The UI reinforces this with an explicit "every metric is computed in Python"
  note on the review card.
- **Heuristic mode is first-class**, not a stub. Target inference, model selection, and review all
  have deterministic fallbacks so the product is fully functional with zero API keys.
- **SSE with backlog replay.** The stream replays history on (re)connect; the client dedupes by a
  stable event key so reconnects don't double-count.
- **Reproducibility.** A single `RANDOM_STATE` seeds splits, CV, and model initialization.
