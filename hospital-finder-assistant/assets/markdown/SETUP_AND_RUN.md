# Setup &amp; Run Guide — Voice Hospital Finder Bot

This guide walks through installing, configuring, and running the project from a clean machine.
For the architecture and design, see [`DOCUMENTATION.md`](DOCUMENTATION.md) and
[`SOURCE_CODE_DOCUMENTATION.md`](SOURCE_CODE_DOCUMENTATION.md).

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11** (pinned to `3.11.14`) | Required — see `pyproject.toml`. |
| **uv** | Modern Python package/venv manager (recommended). `pip` also works. |
| **OpenAI API key** | Used for Whisper STT, embeddings, and TTS. |
| **LLM API key + base URL** | For intent recognition, clarification and RAG grounding (OpenAI / OpenRouter / any OpenAI-compatible endpoint). Default model is `google/gemini-2.0-flash-001`. |
| **Microphone + speakers** | Only needed for **voicebot** mode (PyAudio / ffmpeg). Not required for the default text **chatbot** mode. |
| **(Optional) CUDA GPU** | Recommended if you want to run/serve the QLoRA fine-tuned model. CPU works but is slow. A pre-trained adapter ships under `data/rag_llm/`. |

---

## 2. Clone the repository

```bash
git clone https://github.com/navinpandiyan/hospital-finder-assistant.git
cd hospital-finder-assistant
```

---

## 3. Install dependencies

### Option A — uv (recommended)

```bash
pip install uv      # if you don't already have it
uv sync             # creates .venv/ and installs everything from pyproject.toml / uv.lock
```

### Option B — pip + virtualenv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm   # if the spaCy model isn't pulled automatically
```

> **Note on Torch:** `pyproject.toml` pins CUDA 12.4 Windows wheels for `torch` /
> `torchvision` / `torchaudio`. On a different OS or without a GPU, install the matching
> CPU/GPU build of PyTorch for your platform instead.

---

## 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY="sk-..."                       # Whisper STT, embeddings, TTS
LLM_API_KEY="..."                             # recognition / clarification / RAG grounding
LLM_BASE_URL="https://openrouter.ai/api/v1"   # or https://api.openai.com/v1, etc.
```

> The `.env` file is git-ignored — never commit real keys. Only `.env.example`-style
> templates should be checked in.

---

## 5. Choose your mode and settings (optional)

All runtime behavior lives in [`settings/config.py`](../settings/config.py). The most useful knobs:

| Setting | Default | What it does |
|---------|---------|--------------|
| `MODE` | `"chatbot"` | `"chatbot"` = type your queries; `"voicebot"` = speak and hear replies. |
| `SHOW_LOGS` | `False` | Set `True` to see detailed console logs. |
| `USE_LLM_FOR_RECOGNITION` | `True` | `True` = LLM intent/entity recognition; `False` = spaCy NER + fuzzy matching. |
| `LOOKUP_MODE` | `"rag"` | `"rag"` = FAISS + LLM grounding; `"simple"` = direct DB lookup. |
| `GROUND_WITH_FINE_TUNE` | `True` | Use the local QLoRA model for `find_by_hospital` queries. Set `False` to skip loading it (no GPU needed). |
| `TEXT_TO_DIALOGUE` | `False` | Rewrite factual answers into a warmer spoken style before TTS. |
| `RECOGNIZER_MODEL` / `RAG_GROUNDER_MODEL` | `google/gemini-2.0-flash-001` | Swap in any OpenAI-compatible model. |
| `MAX_TURNS` | `25` | Max clarification turns before giving up. |

> 💡 **Fastest first run / no GPU:** keep `MODE="chatbot"` and set
> `GROUND_WITH_FINE_TUNE=False` to avoid loading the fine-tuned model.

---

## 6. Run the application

```bash
uv run app.py
# or, with an activated venv:
python app.py
```

### What happens on the first run

The app bootstraps everything it needs (idempotent — it skips anything already present):

1. **Seeds the database** — generates 150 synthetic hospitals + 20 insurance plans into `data/hospitals.sqlite`.
2. **Links** hospitals to insurance plans (many-to-many).
3. **Builds the FAISS vector DB** from the hospital records (`data/vdb_hospitals/`).
4. **Generates fine-tuning data** (`data/insurance_data.json`) if missing.
5. **Fine-tunes the QLoRA model** into `data/rag_llm/` *only if that folder doesn't already exist* (a pre-trained adapter is included, so this is normally skipped).
6. **Launches the bot** — you'll see `Bot: Hello! I am your hospital assistant...`

Subsequent runs reuse the existing DB, vector store and model, so they start quickly.

---

## 7. Try it out

Type (or speak, in voicebot mode) queries like:

```text
Find the nearest cardiology hospital in Abu Dhabi
Show me the best dental hospital in Dubai covered by Daman
What insurance does Mleiha ENT Clinic accept?
Compare Mleiha ENT Clinic and Dubai Dental & Obstetrics Care Hub
thanks, that's all      ← ends the conversation
```

Each completed conversation turn is saved as a JSON transcript under `outputs/`
(see the samples already in that folder).

---

## 8. Project layout

```
hospital-finder-assistant/
├── app.py                  # entry point — wires up RAG retriever + LangGraph
├── settings/               # config, prompts, LLM client, logging
├── graphs/                 # LangGraph state machine (nodes, edges) + tools
├── tools/                  # transcribe, recognize, TTS, lookup, RAG retrieve, record
├── db/                     # Pony ORM models, DB bootstrap
│   └── modules/            # synthetic data + vector DB + fine-tune generators
├── data/                   # SQLite DB, FAISS index, fine-tuned QLoRA adapter
├── utils/                  # helpers (geocoding, audio, state save)
├── outputs/                # saved conversation transcripts
└── docs/                   # this guide, documentation, slide deck, diagram
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `spaCy model 'en_core_web_sm' not found` | `python -m spacy download en_core_web_sm` |
| Torch / CUDA install errors | Install the PyTorch build matching your OS/GPU from pytorch.org, then re-run. |
| Slow startup or OOM loading the model | Set `GROUND_WITH_FINE_TUNE = False` in `settings/config.py` (skips the local LLM). |
| No audio / PyAudio errors in voicebot mode | Use `MODE = "chatbot"`, or install PortAudio/ffmpeg and check mic permissions. |
| Auth / 401 errors | Verify `OPENAI_API_KEY`, `LLM_API_KEY` and `LLM_BASE_URL` in `.env`. |

---

*Built October 2025 as an interview case study.*
