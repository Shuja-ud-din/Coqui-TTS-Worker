import runpod
import base64
import tempfile
from runpod.serverless.utils import rp_cuda
from TTS.api import TTS
import torch

print("Initializing Coqui TTS worker...")

# -------------------------
# Device selection
# -------------------------
DEVICE = "cuda" if rp_cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -------------------------
# Model registry
# -------------------------
tts_engines = {}

def load_models():
    """
    Load all TTS models at startup.
    This avoids cold-download latency during requests.
    """

    print("Loading English model -> tts_models/en/ljspeech/vits")
    tts_engines["en"] = TTS(
        model_name="tts_models/en/ljspeech/vits"
    ).to(DEVICE)

    print("Loading Arabic model -> tts_models/ar/mai/tacotron2")
    tts_engines["ar"] = TTS(
        model_name="tts_models/ar/mai/tacotron2"
    ).to(DEVICE)

    print("All models loaded successfully.")

# Load once at container start
load_models()

# -------------------------
# TTS synthesis
# -------------------------
def synthesize(text: str, lang: str):
    if lang not in tts_engines:
        raise ValueError(f"Unsupported language: {lang}")

    tts = tts_engines[lang]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    tts.tts_to_file(text=text, file_path=out_path)

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode("utf-8")

# -------------------------
# RunPod handler
# -------------------------
def handler(event):
    """
    Expected input:
    {
      "input": {
        "text": "Hello world",
        "lang": "en" | "ar"
      }
    }
    """

    input_data = event.get("input", {})
    text = input_data.get("text")
    lang = input_data.get("lang", "en")

    if not text:
        return {"error": "text is required"}

    try:
        audio_base64 = synthesize(text, lang)
    except ValueError as e:
        return {"error": str(e)}

    return {
        "audio": audio_base64,
        "format": "wav",
        "lang": lang
    }

# -------------------------
# Concurrency control
# -------------------------
def adjust_concurrency(current_concurrency):
    # Conservative default for Tacotron2 stability
    return 4 if DEVICE == "cuda" else 1

# -------------------------
# RunPod start
# -------------------------
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "adjust_concurrency": adjust_concurrency
    })
