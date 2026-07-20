FROM python:3.10-slim

WORKDIR /app

# System dependencies: audio I/O (libsndfile, ffmpeg), git (HF/NeMo downloads),
# and a compiler toolchain (several NeMo ASR deps build native extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    build-essential \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch/torchaudio FIRST, from the pytorch CPU index, so that
# nemo_toolkit[asr] (installed next) sees a satisfying torch already present and
# does NOT pull the multi-GB CUDA build.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.12.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu

# Application Python dependencies (torch already satisfied above).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + data:
#   data/quranic_lexicon.json   — Tarteel constrained-decode lexicon (rollback backend)
#   data/curriculum_words.json  — FastConformer curriculum (optional closed-set rescoring)
COPY app.py .
COPY core/ ./core/
COPY data/ ./data/

# torch>=2.6 refuses to load the .nemo checkpoint without this flag.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Pre-download the FastConformer weights at build time so the first request after
# deploy doesn't pull ~0.5GB. Tarteel/legacy weights are NOT pre-baked: they are
# rollback-only backends and would cold-download once if selected. The preferred
# rollback is re-routing traffic to the previous Cloud Run revision (instant).
RUN python -c "import nemo.collections.asr as nemo_asr; \
    nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained('nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0')"

# Pre-download the phoneme MDD recognizer weights (facebook/wav2vec2-xlsr-53-espeak-cv-ft,
# ~1.2GB, Apache-2.0) so the first MDD request doesn't cold-download. Loaded via the same
# components the runtime uses (the Wav2Vec2Processor auto-loader is broken for this 2021
# checkpoint, so we fetch feature-extractor + CTC tokenizer + model directly).
RUN python -c "from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC; \
    m='facebook/wav2vec2-xlsr-53-espeak-cv-ft'; \
    Wav2Vec2FeatureExtractor.from_pretrained(m); \
    Wav2Vec2CTCTokenizer.from_pretrained(m); \
    Wav2Vec2ForCTC.from_pretrained(m)"

# Default to the FastConformer fix. Override with MODEL_VARIANT=legacy (58% wav2vec2,
# cold-downloads once) or MODEL_VARIANT=tarteel (25% Quranic, cold-downloads) for rollback.
ENV MODEL_VARIANT=fastconformer

# Optional: comma-separated backends to warm-load at startup in addition to
# MODEL_VARIANT (e.g. "mdd" so the cascade's phoneme model is ready). Empty = only
# the MODEL_VARIANT backend loads eagerly; others load lazily on first request.
ENV PRELOAD_VARIANTS=""

# Cloud Run injects PORT; default 8080.
EXPOSE 8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
