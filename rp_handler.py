import runpod
import base64
import tempfile
from runpod.serverless.utils import rp_cuda
from TTS.api import TTS
from TTS.utils.manage import ModelManager

print("Initializing Coqui TTS worker...")

DEVICE = "cuda" if rp_cuda.is_available() else "cpu"
print("Using device:", DEVICE)

tts_engines = {}

def load_models():
    print("Loading model -> tts_models/deu/fairseq/vits")
    # Load XTTS v2 by model_name; TTS will handle downloading/cache
    tts = TTS(model_name="tts_models/deu/fairseq/vits").to(DEVICE)
    tts_engines["multi"] = tts
    print("Model loaded successfully.")

load_models()

def synthesize(text: str, lang: str = "multi"):
    tts = tts_engines[lang]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    tts.tts_to_file(text=text, file_path=out_path)

    with open(out_path, "rb") as f:
        audio_bytes = f.read()

    return base64.b64encode(audio_bytes).decode("utf-8")

def handler(event):
    """
    Expected input:
    {
      "input": {
        "text": "Hello world"
      }
    }
    """
    input_data = event["input"]
    text = input_data.get("text")

    if not text:
        return {"error": "text is required"}

    audio_base64 = synthesize(text)

    return {
        "audio": audio_base64,
        "format": "wav",
        "lang": "multi"
    }

def adjust_concurrency(current_concurrency):
    return 10

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "adjust_concurrency": adjust_concurrency
    })
