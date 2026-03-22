# Back-translation similarity (Hugging Face Inference)

This small demo scores **(source, back-translated)** sentence pairs using the **Hugging Face Inference API** sentence-similarity task. It reads pairs from a CSV file, calls the hosted model through **`huggingface_hub`** (which uses the current inference router), prints a sorted table, and runs a couple of **live API sanity checks** so you know scores are not hard-coded.

There are **no local model weights**; everything goes through Hugging Face’s servers.

---

## Prerequisites

- **Python 3.11+** (3.12 is fine)
- A **Hugging Face account** and an **API token** with permission to call Inference ([create a token](https://huggingface.co/settings/tokens))

---

## After you clone the repo

### 1. Go into the project directory

```bash
cd hack-demo-prep
```

(Use the path where you cloned the repository.)

### 2. Create and activate a virtual environment

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set your Hugging Face API token

The script reads **`HF_API_TOKEN`** from the environment. Recommended for local work: a **`.env`** file in the **project root** (same folder as `evaluate_backtranslation.py`). The file is **gitignored** so your token is not committed.

Create `.env`:

```env
HF_API_TOKEN=hf_your_token_here
```

Alternatively, export it in your shell for the current session:

```bash
export HF_API_TOKEN=hf_your_token_here
```

### 5. Run the evaluator

From the project root, with the venv activated:

```bash
python evaluate_backtranslation.py
```

By default this uses **`es_ar_backtranslated_sample_20.csv`** in the current directory. To use another file:

```bash
python evaluate_backtranslation.py path/to/your.csv
```

To save output to a file while still seeing it in the terminal:

```bash
python evaluate_backtranslation.py | tee results.txt
```

---

## CSV format

The loader expects a header row with:

- **`Source`** (or any column that normalizes to `source`)
- **`Back-translated`** (or any column that normalizes to `backtranslated`)

An optional third column **`Score`** (e.g. scores from another system) is shown in the table as **`gemini`** when present.

Both text columns must be non-empty on each data row.

---

## Optional environment variables

| Variable | Purpose |
|----------|---------|
| **`HF_API_TOKEN`** | Required. Your Hugging Face API token. |
| **`HF_SIMILARITY_MODEL_ID`** | Optional. Hub model id for sentence similarity (default: `sentence-transformers/all-MiniLM-L6-v2`). |

---

## What you should see

- A line with **average** HF similarity (rounded for display).
- A **fixed-width table**: `idx`, `similarity`, `source_snippet`, `backtranslation_snippet` (and **`gemini`** if the CSV has scores).
- **Sanity checks** that call the real API twice (identical vs unrelated strings), then a short **success** line if both pass.

If `HF_API_TOKEN` is missing or empty, the script exits with a clear error telling you to use `.env` or `export`.

---

## Troubleshooting

- **HTTP errors or timeouts:** The model may be loading on Hugging Face’s side; retry after a short wait. Check that your token is valid and has not expired.
- **Rate limits:** Reduce batch size or wait between runs if you hit provider limits.
- **Wrong Python:** Use `python3` and ensure the venv is activated (`which python` should point inside `.venv`).

---

## Project layout (important files)

| File | Role |
|------|------|
| `evaluate_backtranslation.py` | Main script |
| `requirements.txt` | Python dependencies |
| `es_ar_backtranslated_sample_20.csv` | Example input CSV |
| `.env` | Your secrets (create locally; not in git) |
