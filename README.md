# Enterprise Document Assistant

A Retrieval-Augmented Generation system that answers natural-language questions
about internal company documents — policies, handbooks, manuals, reports — and
cites the file and page every claim came from.

```
PDFs → text extraction → chunking → embeddings → Chroma vector store
     → semantic retrieval → LLM constrained to the retrieved context
     → grounded answer + citations
```

The point of the system is not that it can answer questions. It is that it
**refuses to answer the ones your documents don't cover**, instead of inventing
a plausible policy number.

---

## Quickstart

> **Needs Python 3.11 or newer.** On macOS, bare `python3` is still the system
> 3.9, which cannot install these dependencies (numpy needs 3.12+, pandas
> 3.11+, streamlit 3.10+). Check with `python3 -V`; if it prints 3.9, run
> `brew install python@3.12` first. `make setup` picks the right interpreter
> for you.

```bash
make setup                    # creates .venv on a suitable Python, installs deps, copies .env
make sample                   # writes 4 sample enterprise PDFs
make index                    # parse → chunk → embed → index
make ui                       # Streamlit at http://localhost:8501
```

<details>
<summary>Without <code>make</code></summary>

```bash
python3.12 -m venv .venv       # NOT `python3` on macOS — that is 3.9
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m scripts.generate_test_pdfs
python -m scripts.ingest --rebuild
streamlit run app.py
```
</details>

Ask from the terminal instead of the browser:

```bash
.venv/bin/python -m scripts.ask "How many days of paid time off do employees get?"
.venv/bin/python -m scripts.ask                 # interactive loop
```

**No API key?** Everything except answer synthesis is local and free. Set
`LLM_PROVIDER=extractive` in `.env` and the whole pipeline runs with no
credentials and no network — see [Providers](#providers) below. Without either
a key or that setting, the assistant tells you so in one line rather than
raising.

---

## What each stage does, and why

### 1. Ingestion — `docassist/loader.py`

`PyPDFLoader` reads each PDF one page at a time, and every extracted page keeps
its filename and page number. That metadata is not decoration: it is the only
reason a citation at the far end of the pipeline can be checked by a human.

PDF text extraction is messy — words get split across line breaks
(`confiden-\ntial`), and sentences arrive with hard newlines in the middle.
Both are normalised here, because both degrade chunking and embedding quality.
Pages that extract to almost nothing (cover sheets, scanned images) are dropped
rather than indexed as noise.

### 2. Chunking — `docassist/chunker.py`

`RecursiveCharacterTextSplitter`, `chunk_size=1000`, `chunk_overlap=200`.

A whole page is the wrong unit for retrieval. Embed a page about leave,
overtime *and* sick pay and you get a vector that is precisely about none of
them. Split too finely and a fact loses the context that makes it meaningful.
~1000 characters keeps roughly one idea per vector.

The 200-character overlap exists for a specific failure: a fact that straddles a
chunk boundary — "…must be reported within" / "24 hours to the Service Desk" —
would otherwise be unretrievable in either half. Overlap guarantees it appears
whole in at least one chunk.

The splitter tries paragraph breaks first, then line breaks, then sentence
endings, and only cuts mid-word as a last resort.

Each chunk gets a **stable id** derived from its content and origin, so
re-running ingestion upserts rather than duplicating. Adding one new PDF to a
corpus costs one PDF's worth of embedding, not the whole corpus.

### 3. Embeddings — `docassist/embeddings.py`

Default: `sentence-transformers/all-MiniLM-L6-v2`, running locally on CPU. Free,
offline after the first download, and good enough that retrieval quality is not
the bottleneck on documents this size.

Embeddings are what make this semantic rather than keyword search. In the sample
corpus, the handbook never uses the phrase "paid time off" — it says "annual
leave" throughout. Asking *"how many days of paid time off do I get?"* still
retrieves the right chunk, at cosine similarity 0.55. Ctrl-F cannot do that.

Vectors are L2-normalised, and the Chroma collection is built in **cosine
space** rather than the default L2, so the relevance score LangChain returns is
literally cosine similarity — a number you can set a threshold against and
reason about, rather than an artefact of the distance metric.

### 4. Vector storage — `docassist/vectorstore.py`

ChromaDB, persisted to `storage/chroma/`. Stores the chunk text, its embedding
and its metadata, and does the nearest-neighbour search. Runs in-process with no
server to operate, which is the right trade for a corpus of this size.

### 5. Retrieval — `docassist/retriever.py`

The question is embedded with **the same model used for indexing** (vectors from
two different models are not comparable — the search silently returns nonsense
rather than erroring), compared against every stored chunk, and the top `k`
come back.

Chunks below `SCORE_THRESHOLD` are discarded. Overlapping chunks that duplicate
the same passage are collapsed. The survivors are formatted into a numbered
block — `[1] Source: handbook.pdf | Page: 3` — and it is that numbering that
makes citation possible at all.

Only these few chunks reach the LLM, not the corpus. That is what keeps the
prompt small, the cost low, and the answer focused on the passage that matters.

`SEARCH_TYPE=mmr` swaps plain similarity for maximal marginal relevance, which
trades a little relevance for more diverse chunks — better for *"summarise this
document"*, where plain similarity tends to return five near-identical chunks
from the same section.

### 6. Answer synthesis — `docassist/prompts.py`, `docassist/llm.py`

Retrieval decides what the model is *allowed to see*. The prompt decides what it
is *allowed to do with it*:

- answer using only the numbered passages — general knowledge is not a valid
  source here, even when it happens to be correct;
- cite the passage behind each claim as `[1]`, `[2]`;
- if the context doesn't answer the question, say exactly
  *"The provided documents do not contain this information."*;
- if it answers only partly, give that part and say what is missing;
- never invent policy numbers, dates, names or amounts;
- if two passages conflict, report both rather than picking one.

### 7. Verification — `docassist/pipeline.py`

Two checks that go beyond a textbook RAG loop, because a prompt is a request,
not a guarantee:

**Abstention before generation.** If no chunk clears the relevance threshold,
the LLM is never called at all. A model handed irrelevant context will still try
to be helpful, and that is precisely where fabrication comes from. The cheapest
hallucination to prevent is the one you never generate.

**Citation verification after generation.** Every `[n]` the model writes is
checked against the passages it was actually given. A citation pointing at a
passage that does not exist is a fabrication wearing a uniform, and it is
flagged in red in the UI. The pipeline also reports what fraction of the
answer's factual lines carry a citation at all.

`response.grounded` is true only when the answer is either a clean abstention or
fully and validly cited.

### 8. Interface — `app.py`

Streamlit. Upload PDFs, index them, ask questions; the answer renders with
expandable source cards showing the file, page, relevance score and the actual
snippet the claim rests on. Footer shows model, latency, token usage and
citation coverage.

---

## Measuring it

The claim "RAG reduces hallucination" is only worth making if you measure it.

```bash
python -m scripts.evaluate                          # with your configured LLM
python -m scripts.evaluate --provider extractive    # retrieval-only baseline
```

`scripts/evaluate.py` scores two sets that pull in opposite directions:

- **10 answerable questions** — did it find the right fact, from the right file,
  and cite it?
- **6 unanswerable questions** — did it refuse?

The second set is the one that matters. Any system looks good on questions its
documents answer; a hallucinating system is one that *also* answers the ones
they don't. Three of the six are adversarial: they are topically adjacent to
indexed content (parental leave, Q4 revenue, disciplinary penalties), so
retrieval cheerfully returns plausible-looking chunks and only the grounding
layer stops the model from filling in the blank.

Measured on the sample corpus with the **retrieval-only baseline**
(`--provider extractive`, no model in the loop):

| Metric | Score |
|---|---|
| Answer accuracy | 8/10 (80%) |
| Correct source attribution | 10/10 (100%) |
| Refusal rate on unanswerable | **2/6 (33%)** |

That 33% is the finding, not a failure. Retrieval alone cannot abstain — it
always returns *something*, and something always looks like an answer. Only the
two wholly off-topic questions fall below the relevance threshold. Closing the
remaining gap is exactly what the grounding prompt and the citation check are
for; re-run without `--provider extractive` to measure that difference on your
own key.

---

## Providers

Configured in `.env`. Embeddings and LLM are chosen independently.

| `LLM_PROVIDER` | What it uses | Needs |
|---|---|---|
| `anthropic` *(default)* | Claude via the Messages API | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI chat models | `OPENAI_API_KEY`, `pip install langchain-openai` |
| `ollama` | a local model via Ollama | `ollama serve` running |
| `extractive` | **no model at all** | nothing |

`extractive` stitches an answer from the highest-scoring sentences in the
retrieved passages. It is not fluent, but it is physically incapable of emitting
a word that isn't in your documents — which makes it a hard floor on
hallucination, a way to exercise the retrieval half of the pipeline with zero
credentials, and something deterministic for the test suite to assert against.

| `EMBEDDING_PROVIDER` | Model | Needs |
|---|---|---|
| `huggingface` *(default)* | `all-MiniLM-L6-v2`, local CPU | nothing after first download |
| `openai` | `text-embedding-3-small` | `OPENAI_API_KEY`, `pip install langchain-openai` |

> Changing the embedding model invalidates the index — the old vectors were
> produced by a different model and are not comparable. Re-run
> `python -m scripts.ingest --rebuild`.

---

## Configuration

Every knob lives in `.env` (see `.env.example`); every one has a working
default. `docassist/config.py` is the only module that reads the environment.

| Variable | Default | Notes |
|---|---|---|
| `CHUNK_SIZE` | `1000` | characters per chunk |
| `CHUNK_OVERLAP` | `200` | must be < `CHUNK_SIZE` |
| `TOP_K` | `5` | chunks retrieved per question |
| `SEARCH_TYPE` | `similarity` | or `mmr` for diversity |
| `SCORE_THRESHOLD` | `0.15` | minimum cosine similarity to be shown to the LLM |
| `MAX_CONTEXT_CHARS` | `12000` | hard cap on prompt context |
| `LLM_EFFORT` | `medium` | Claude reasoning effort — grounded extraction doesn't need `high` |
| `LLM_MAX_TOKENS` | `4096` | |

On the sample corpus a strong match scores 0.4–0.7, a weak-but-real match ~0.25,
and a genuinely off-topic question tops out below 0.10 — so `0.15` keeps recall
while still refusing what the documents cannot answer. Raise it to make the
assistant more willing to say "I don't know".

---

## Layout

```
docassist/
  config.py        settings; the only module that reads the environment
  loader.py        stage 1 — PDF → page Documents with source metadata
  chunker.py       stage 2 — page → overlapping chunks with stable ids
  embeddings.py    stage 3 — pluggable embedding model
  vectorstore.py   stage 4 — Chroma persistence + index stats
  retriever.py     stage 5 — semantic search, threshold filter, context block
  prompts.py       the grounding contract
  llm.py           stage 6 — provider adapters behind one interface
  pipeline.py      orchestration + abstention + citation verification
  ingest.py        stages 1–4 as one call
scripts/
  generate_test_pdfs.py   4 realistic sample documents
  ingest.py               build/refresh/inspect the index
  ask.py                  terminal Q&A, one-shot or interactive
  evaluate.py             the grounding benchmark
tests/                    24 tests, all offline
app.py                    Streamlit UI
```

---

## Tests

```bash
python -m pytest tests/ -q
```

24 tests, no network and no API key. They use a stub LLM, because what needs
testing is the pipeline's behaviour *around* the model — that it abstains
without calling out, that it filters low-relevance chunks, that it lists only
the sources actually cited, that it flags a citation pointing at a passage that
was never retrieved — not the model's prose. `tests/test_anthropic_provider.py`
asserts the exact request shape sent to the Anthropic API against a mocked
client.

---

## The sample corpus

`scripts/generate_test_pdfs.py` writes four documents for a fictional logistics
company, built to exercise specific behaviours rather than just fill pages:

| Document | Contains |
|---|---|
| `Northwind_Employee_Handbook_2026.pdf` | working hours, annual leave, sick leave, remote work, conduct |
| `Northwind_Data_Privacy_Policy.pdf` | GDPR basis, retention periods, data-subject rights, breach notification |
| `Northwind_IT_Security_Policy.pdf` | access control, passwords/MFA, devices, incident response, AI tool use |
| `Northwind_Q3_2025_Operations_Report.pdf` | volume, revenue, cost, service quality, headcount, outlook |

Deliberately built in:

- **vocabulary mismatch** — "annual leave" is never called "paid time off", so
  only semantic search finds it;
- **cross-document overlap** — data retention appears in both the privacy and
  security policies, so retrieval has to pick the right one;
- **real gaps** — nothing covers parental leave pay, stock options, Q4 revenue
  or the CEO's name, which is what the abstention tests probe.
