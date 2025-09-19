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

# --- NEW: Supabase client
from supabase import create_client, Client

# Initialize FastAPI
app = FastAPI()

# Class names in the required order (0, 1, 2, 3, 4)
CLASS_NAMES = ["apple", "banana", "paprika", "orange", "tomato"]

# Cloud storage URLs for related images (static references)
CLOUD_STORAGE_URLS = {
    0: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/red-apple.jpg",
    1: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/yellow-banana.jpg",
    2: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/green-paprika.jpg",
    3: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/orange.jpg",
    4: "https://biennddgojeyshruogja.supabase.co/storage/v1/object/public/learning-assets/tomato.jpg",
}

# --- NEW: Supabase envs and client
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://biennddgojeyshruogja.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # set in Railway
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "Upload-Bucket")

_supabase: Client | None = None

def get_supabase() -> Client | None:
    global _supabase
    if _supabase is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            _supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("[SUPABASE] client initialized")
        except Exception as e:
            print(f"[SUPABASE] init failed: {e}")
            _supabase = None
    return _supabase

# Path to the model file
MODEL_PATH = "buah.h5"  # Ensure this is copied to the container's working directory

# Load TensorFlow model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Initialize ImageClassifier
image_classifier = ImageClassifier(model)

@app.get("/")
def index():
    """
    Root endpoint to check if the API is running properly.
    """
    return {"message": "Cunny Fruits Prediction API (apple, banana, paprika, orange, tomato) is running"}

# --- NEW: helper to upload bytes to Supabase and return a public URL
def _upload_to_supabase(file_bytes: bytes, filename: str) -> str | None:
    """
    Uploads bytes to Supabase Storage (SUPABASE_BUCKET) and returns a public URL.
    Returns None if Supabase is not configured.
    """
    if supabase is None:
        return None
    # Optional: prefix with a folder and a simple unique suffix
    # Avoid collisions, keep it simple without changing your structure
    base_name = os.path.basename(filename)
    key = f"uploads/{base_name}"

    # upsert=True to overwrite on same name (you can switch to unique names if needed)
    res = supabase.storage.from_(SUPABASE_BUCKET).upload(key, file_bytes, {"upsert": True})
    if isinstance(res, dict) and res.get("error"):
        # Log silently; don’t kill the request just because upload failed
        print("Supabase upload error:", res["error"])
        return None

    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(key)
    return public_url

@app.post("/predict")
async def predict_image(uploaded_file: UploadFile, response: Response):
    """
    API for predicting the fruit based on an uploaded image (0 = apple, 1 = banana, 2 = paprika, 3 = orange, 4 = tomato).
    Returns the predicted class name and the corresponding image URL from cloud storage.
    Also uploads the received image to Supabase (if configured) and returns its public URL.
    """
    try:
        # Checking if it's an image
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return {"error": True, "message": "File is not an image"}

        # Read file content (async to avoid blocking the event loop)
        file_content = await uploaded_file.read()

        # Check file size (10 MB = 10 * 1024 * 1024 bytes)
        if len(file_content) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "message": "Cannot upload files more than 10MB, try another image"
                }
            )

        # Load image for prediction (force RGB to avoid mode issues)
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # Predict using the ImageClassifier
        predicted_class, _ = image_classifier.load_and_predict(image)

        # Convert predicted class to a Python int (to be JSON serializable)
        predicted_class = int(predicted_class)

        # Get the class name and image URL
        predicted_label = CLASS_NAMES[predicted_class]
        image_url = CLOUD_STORAGE_URLS.get(predicted_class)

        # --- NEW: upload the user's image to Supabase (optional, if configured)
        uploaded_public_url = _upload_to_supabase(file_content, uploaded_file.filename)

        payload = {
            "error": False,
            "message": "Prediction successfully !",
            "prediction_result": {
                "predicted_class": predicted_class,
                "predicted_label": predicted_label,
                "reference_image_url": image_url,      # your static reference image
                "uploaded_image_url": uploaded_public_url,  # the actual uploaded file (if available)
            }
        }

        return JSONResponse(status_code=200, content=payload)

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return {"error": True, "message": f"Prediction failed: {e}"}

# Starting the server
# You can check the API documentation easily using /docs after the server is running
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
