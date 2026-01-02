import runpod
import base64
import tempfile
from runpod.serverless.utils import rp_cuda
from TTS.api import TTS

print("Initializing Coqui TTS worker...")

DEVICE = "cuda" if rp_cuda.is_available() else "cpu"

MODELS = {
    "en": "tts_models/en/vctk/vits",
    "ar": "tts_models/ar/mai/tacotron2-DDC"
}

tts_engines = {}

def load_models():
    for lang, model_name in MODELS.items():
        print(f"Loading model: {lang} -> {model_name}")
        tts_engines[lang] = TTS(
            model_name=model_name,
            gpu=(DEVICE == "cuda")
        )

load_models()
print("Models loaded successfully.")

def synthesize(text: str, lang: str):
    if lang not in tts_engines:
        raise ValueError("Unsupported language")

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
        "text": "Hello world",
        "lang": "en"
      }
    }
    """
    input_data = event["input"]

    text = input_data.get("text")
    lang = input_data.get("lang", "en")

    if not text:
        return {"error": "text is required"}

    audio_base64 = synthesize(text, lang)

    return {
        "audio": audio_base64,
        "format": "wav",
        "lang": lang
    }

def adjust_concurrency(current_concurrency):
    return 10

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler, 
        "adjust_concurrency": adjust_concurrency
    })
