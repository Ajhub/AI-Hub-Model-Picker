# AI Hub Model Picker

Describe an AI use case in plain English and get the best-matching models
from a local catalog of Qualcomm AI Hub models, ranked by semantic
similarity, with example latency and power figures for each.

Example: *"transcribe and summarize customer support calls"* returns
Whisper (speech-to-text) and Llama (summarisation) models.

The matching runs **entirely on your machine**. After the first run has
downloaded the small embedding model, no internet connection is needed and
no query or data leaves the device.

## How it works

1. `catalog.json` holds ~37 AI Hub models (name, category, description,
   example latency, power note).
2. `embed_utils.py` embeds each description once with `all-MiniLM-L6-v2`
   and caches the vectors in `embeddings.npy`.
3. When you search, your query is embedded and compared to every model with
   cosine similarity. The top 5 are shown in `app.py` (Streamlit).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `embed_utils.py` | Embedding, caching and ranking logic |
| `catalog.json` | The model catalog |
| `requirements.txt` | Python dependencies |

## Install

Requires Python 3.9+. No GPU needed.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Optional: pre-compute the catalog embeddings (the app also does this
# automatically on first launch)
python embed_utils.py

streamlit run app.py
```

Streamlit opens the app in your browser, usually at http://localhost:8501.

The **first run needs internet once** to download the embedding model
(~90 MB, cached afterwards). Later runs work with Wi-Fi off.

## Important: catalog figures are examples

Model names and descriptions are representative of what is available on
Qualcomm AI Hub, but **the latency and power values are placeholder example
figures**, not measurements. Before submitting, verify the model names
against https://aihub.qualcomm.com and replace the numbers with real
profiling results from AI Hub Workbench.

## Porting to Snapdragon NPU

The app currently embeds queries on CPU with `sentence-transformers`. The
"CPU vs NPU" panel is a stand-in until the model runs on the Hexagon NPU.
To make the matching engine genuinely Snapdragon-optimized:

1. **Install the AI Hub SDK**: `pip install qai-hub`, create an account at
   https://aihub.qualcomm.com and configure your API token
   (`qai-hub configure --api_token <TOKEN>`). This works from any machine,
   including one without a Snapdragon chip.
2. **Export the embedding model to ONNX**: export
   `sentence-transformers/all-MiniLM-L6-v2` (for example with
   `optimum-cli export onnx` or `torch.onnx.export`) using fixed input
   shapes, e.g. a max sequence length of 128.
3. **Compile for a Snapdragon target with qai-hub**: use
   `qai_hub.get_devices()` to pick a Snapdragon X Elite / X Plus device,
   then `qai_hub.submit_compile_job(...)` on your ONNX model. Optionally
   submit a profile job to get real latency numbers on real hardware.
4. **Run with the QNN execution provider in ONNX Runtime**: on the
   Snapdragon HP PC, install an ONNX Runtime build that includes the QNN
   execution provider and create the session with
   `providers=["QNNExecutionProvider"]` (backend: the HTP / Hexagon NPU
   library), so inference runs on the NPU.
5. **Swap the backend in `embed_utils.py`**: replace the
   `sentence-transformers` call in `embed_query()` with tokenization
   (keep the Hugging Face tokenizer) + the ONNX Runtime QNN session +
   mean pooling and normalisation, matching what `SentenceTransformer`
   does internally.
6. **Update the timing panel in `app.py`**: replace the "Default backend"
   bar with the QNN (NPU) measurement. Search for the `TODO` comment.
7. **Replace placeholder figures**: put the real profiling numbers from
   step 3 into `catalog.json` (`latency_ms`, `power_note`).

Verify each step against the current Qualcomm AI Hub and ONNX Runtime QNN
documentation, as tooling changes quickly.
