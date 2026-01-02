import runpod
import base64
import tempfile
from runpod.serverless.utils import rp_cuda
from TTS.api import TTS

print("Initializing Coqui TTS worker...")

DEVICE = "cuda" if rp_cuda.is_available() else "cpu"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_NAME}")

tts = TTS(
    model_name=MODEL_NAME,
    gpu=(DEVICE == "cuda")
)

print("XTTS model loaded successfully.")

SUPPORTED_LANGS = {"en", "ar"}

def synthesize(text: str, lang: str):
    if lang not in SUPPORTED_LANGS:
        raise ValueError("Unsupported language")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    # XTTS supports language parameter directly
    tts.tts_to_file(
        text=text,
        file_path=out_path,
        language=lang
    )

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode("utf-8")

def handler(event):
    """
    Expected input:
    {
      "input": {
        "text": "Hello world",
        "lang": "en"
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
    except Exception as e:
        return {"error": str(e)}

    return {
        "audio": audio_base64,
        "format": "wav",
        "lang": lang,
        "model": MODEL_NAME
    }

def adjust_concurrency(current_concurrency):
    # Safe default for XTTS on GPU
    return 5

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "adjust_concurrency": adjust_concurrency
    })
