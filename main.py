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

# Initialize FastAPI
app = FastAPI()

# Class names in the required order (0, 1, 2, 3, 4)
CLASS_NAMES = ["apple", "banana", "paprika", "orange", "tomato"]

# Cloud storage URLs for related images
CLOUD_STORAGE_URLS = {
    0: "https://storage.googleapis.com/cunny-assets-bucket/prediction_result_images/red-apple.jpg",  
    1: "https://storage.googleapis.com/cunny-assets-bucket/prediction_result_images/yellow-banana.jpg",   
    2: "https://storage.googleapis.com/cunny-assets-bucket/prediction_result_images/green-paprika.jpg", 
    3: "https://storage.googleapis.com/cunny-assets-bucket/prediction_result_images/orange.jpg",
    4: "https://storage.googleapis.com/cunny-assets-bucket/prediction_result_images/tomato.jpg",
}

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
    # return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.post("/predict")
def predict_image(uploaded_file: UploadFile, response: Response):
    """
    API for predicting the fruit based on an uploaded image (0 = apple, 1 = banana, 2 = paprika, 3 = orange, 4 = tomato).
    Returns the predicted class name and the corresponding image URL from cloud storage.
    """
    try:
        # Checking if it's an image
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return {"error": True, "message": "File is not an image"}

        # Read file content
        file_content = uploaded_file.file.read()

        # Check file size (10 MB = 10 * 1024 * 1024 bytes)
        if len(file_content) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=413,
                content={
                    "error": True,
                    "message": "Cannot upload files more than 10MB, try another image"
                }
            )

        # Load image for prediction
        image = Image.open(io.BytesIO(file_content))

        # Predict using the ImageClassifier
        predicted_class, _ = image_classifier.load_and_predict(image)

        # Convert predicted class to a Python int (to be JSON serializable)
        predicted_class = int(predicted_class)

        # Get the class name and image URL
        predicted_label = CLASS_NAMES[predicted_class]
        image_url = CLOUD_STORAGE_URLS.get(predicted_class)

        return JSONResponse(
            status_code=200,
            content={
                "error": False,
                "message": "Prediction successfully !",
                "prediction_result": {
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_label,
                    "image_url": image_url,
                }
            }
        )

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
