# image_classifier.py

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

class ImageClassifier:
    def __init__(self, model):
        """
        Initialize the ImageClassifier with a trained model.
        """
        self.model = model

    def load_and_predict(self, pil_image):
        """
        Preprocess an image and make a prediction using the model.
        Returns the predicted class and the preprocessed image.
        """
        # Resize and preprocess the image
        pil_image = pil_image.resize((200, 200))  # Resize to match model input
        img_array = img_to_array(pil_image)  # Convert to numpy array
        img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize to [0, 1]

        # Make a prediction
        predictions = self.model.predict(img_array)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])
        return predicted_class, img_array[0]  # Return predicted class and preprocessed image
