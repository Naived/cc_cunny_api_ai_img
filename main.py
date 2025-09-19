import os
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from image_classifier import ImageClassifier

# --- Supabase client (lazy init + non-fatal)
from supabase import create_client, Client

app = FastAPI()

# Classes (0..4)
CLASS_NAMES = ["apple", "banana", "paprika", "orange", "tomato"]

# Static reference images
CLOUD_STORAGE_URLS = {
    0: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/red-apple.jpg",
    1: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/yellow-banana.jpg",
    2: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/green-paprika.jpg",
    3: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/orange.jpg",
    4: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/tomato.jpg",
}

# --- Supabase envs
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://biennddgojeyshruogja.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # set in Railway
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "Upload-Bucket")

_supabase: Client | None = None

def get_supabase() -> Client | None:
    """Lazy init Supabase so bad keys/versions don't crash the app."""
    global _supabase
    if _supabase is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("[SUPABASE] client initialized")
        except Exception as e:
            print(f"[SUPABASE] init failed: {e}")
            _supabase = None
    return _supabase

# --- Model load (lazy + legacy H5 compatibility)
from tensorflow.keras.layers import InputLayer as KInputLayer
from tensorflow.keras import mixed_precision

MODEL_PATH = os.getenv("MODEL_PATH", "buah.h5")

_model = None
_image_classifier = None
_startup_error = None

class CompatibleInputLayer(KInputLayer):
    """Shim to map legacy 'batch_shape' -> 'batch_input_shape' during H5 deserialization."""
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super().__init__(*args, **kwargs)

def DTypePolicy(**kwargs):
    """
    Shim for legacy-serialized dtype policies like:
    {'module': 'keras', 'class_name': 'DTypePolicy', 'config': {'name': 'float32'}}
    """
    name = kwargs.get("name", "float32")
    return mixed_precision.Policy(name)

def _load_model_once():
    """Load the model exactly once; never crash the process if it fails."""
    global _model, _image_classifier, _startup_error
    if _model is not None:
        return
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "InputLayer": CompatibleInputLayer,
                "DTypePolicy": DTypePolicy,  # <-- important
            },
        )
        _image_classifier = ImageClassifier(_model)
        print("[BOOT] Model loaded OK:", MODEL_PATH)
    except Exception as e:
        _startup_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

@app.on_event("startup")
def on_startup():
    # Try to load at boot; if it fails, app still stays up and /healthz reports the reason.
    _load_model_once()

@app.get("/")
def index():
    return {"message": "Cunny Fruits Prediction API (apple, banana, paprika, orange, tomato) is running"}

@app.get("/healthz")
def healthz():
    """Basic health/status including model state."""
    return {
        "ok": _startup_error is None,
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "error": _startup_error,
    }

# --- Upload helper: send bytes to Supabase Storage and return public URL
def _upload_to_supabase(file_bytes: bytes, filename: str) -> str | None:
    client = get_supabase()
    if client is None:
        return None
    key = f"uploads/{os.path.basename(filename)}"
    res = client.storage.from_(SUPABASE_BUCKET).upload(key, file_bytes, {"upsert": True})
    if isinstance(res, dict) and res.get("error"):
        print("Supabase upload error:", res["error"])
        return None
    return client.storage.from_(SUPABASE_BUCKET).get_public_url(key)

@app.post("/predict")
async def predict_image(uploaded_file: UploadFile, response: Response):
    """
    Predict the fruit (0=apple,1=banana,2=paprika,3=orange,4=tomato).
    Returns predicted class/label, a static reference image URL, and (optionally) the uploaded image's public URL in Supabase.
    """
    try:
        # Ensure model is loaded
        if _image_classifier is None:
            _load_model_once()
        if _image_classifier is None:
            response.status_code = 500
            return {"error": True, "message": f"Model unavailable: {_startup_error}"}

        # Validate content type
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return {"error": True, "message": "File is not an image (jpeg or png required)"}

        # Read file content (async)
        file_content = await uploaded_file.read()

        # 10 MB size guard
        if len(file_content) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={"error": True, "message": "Cannot upload files more than 10MB, try another image"},
            )

        # Load image (force RGB)
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # Predict
        predicted_class, _ = _image_classifier.load_and_predict(image)
        predicted_class = int(predicted_class)

        predicted_label = CLASS_NAMES[predicted_class]
        image_url = CLOUD_STORAGE_URLS.get(predicted_class)

        # Optional: upload the user's image to Supabase
        uploaded_public_url = _upload_to_supabase(file_content, uploaded_file.filename)

        return JSONResponse(
            status_code=200,
            content={
                "error": False,
                "message": "Prediction successfully !",
                "prediction_result": {
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_label,
                    "reference_image_url": image_url,
                    "uploaded_image_url": uploaded_public_url,
                },
            },
        )

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": True, "message": f"Prediction failed: {e}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
