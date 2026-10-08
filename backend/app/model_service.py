"""
Model Service Module for Handwritten Digit Recognition.

Handles trained Keras model loading, MNIST-style image preprocessing, and inference.
"""

import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf


class ModelService:
    def __init__(self, model_path: str = None):
        """
        Initialize ModelService with path to trained Keras model artifact.
        """
        if model_path is None:
            model_path = os.getenv("MODEL_PATH")

        if model_path is None:
            # Resolve path relative to project root: model/saved_models/handwritten_digit_model.keras
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            model_path = os.path.join(base_dir, "model", "saved_models", "handwritten_digit_model.keras")

        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Load the trained Keras model into memory on application startup.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model artifact not found at: {self.model_path}")
        print(f"Loading trained Keras model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        print("Keras model loaded successfully.")

    def preprocess_image(self, image_bytes: bytes, save_debug: bool = False) -> np.ndarray:
        """
        Preprocess uploaded raw image bytes into an MNIST-style (1, 28, 28) normalized float32 array.

        MNIST-Style Steps:
        1. Decode raw bytes to PIL Grayscale ('L' mode).
        2. Convert to NumPy float32 array.
        3. Detect and invert background (ensuring background is 0, digit is > 0).
        4. Apply noise thresholding (pixels <= 30 set to 0).
        5. Find non-zero bounding box around the digit.
        6. Crop digit and scale while preserving aspect ratio into a 20x20 bounding box.
        7. Place scaled 20x20 digit into the center of a blank 28x28 canvas.
        8. Shift by intensity Center of Mass so digit is centered like MNIST training set.
        9. Rescale pixel values to range [0.0, 1.0].
        10. Expand batch dimension to shape (1, 28, 28).
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise ValueError(f"Failed to decode image file: {str(e)}")

        # 1 & 2. Convert to Grayscale float32 numpy array
        img_gray = img.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        # 3. Detect background color & ensure background = 0, foreground = positive
        if np.mean(img_array) > 127.0:
            img_array = 255.0 - img_array

        # 4. Threshold background noise
        img_array[img_array < 30.0] = 0.0

        # 5. Find bounding box around digit pixels
        active_pixels = np.argwhere(img_array > 30.0)

        # If canvas is empty or contains no digit strokes above threshold
        if active_pixels.size == 0:
            return np.zeros((1, 28, 28), dtype=np.float32)

        ymin, xmin = active_pixels.min(axis=0)
        ymax, xmax = active_pixels.max(axis=0)

        # 6. Crop digit from bounding box
        digit_crop = img_array[ymin:ymax + 1, xmin:xmax + 1]
        crop_h, crop_w = digit_crop.shape

        # 7. Scale digit to fit inside a 20x20 box preserving aspect ratio
        target_max_size = 20.0
        scale = target_max_size / max(crop_h, crop_w)
        new_h = max(1, int(round(crop_h * scale)))
        new_w = max(1, int(round(crop_w * scale)))

        crop_pil = Image.fromarray(digit_crop.astype(np.uint8))
        resized_pil = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_array = np.array(resized_pil, dtype=np.float32)

        # 8. Create empty 28x28 canvas (black background = 0) & center resized digit
        canvas_28 = np.zeros((28, 28), dtype=np.float32)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_array

        # 9. Center of Mass shift to match MNIST dataset distribution
        total_intensity = np.sum(canvas_28)
        if total_intensity > 0:
            y_indices, x_indices = np.indices((28, 28))
            cy = np.sum(y_indices * canvas_28) / total_intensity
            cx = np.sum(x_indices * canvas_28) / total_intensity

            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))

            # Apply shift within 28x28 bounds
            if shift_y != 0 or shift_x != 0:
                shifted_canvas = np.zeros((28, 28), dtype=np.float32)

                src_y_start = max(0, -shift_y)
                src_y_end = min(28, 28 - shift_y)
                dst_y_start = max(0, shift_y)
                dst_y_end = min(28, 28 + shift_y)

                src_x_start = max(0, -shift_x)
                src_x_end = min(28, 28 - shift_x)
                dst_x_start = max(0, shift_x)
                dst_x_end = min(28, 28 + shift_x)

                h_slice = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
                w_slice = min(src_x_end - src_x_start, dst_x_end - dst_x_start)

                if h_slice > 0 and w_slice > 0:
                    shifted_canvas[dst_y_start:dst_y_start + h_slice, dst_x_start:dst_x_start + w_slice] = \
                        canvas_28[src_y_start:src_y_start + h_slice, src_x_start:src_x_start + w_slice]
                    canvas_28 = shifted_canvas

        # 10. Optional debug image export
        if save_debug or os.getenv("SAVE_DEBUG_IMAGES", "false").lower() == "true":
            try:
                debug_dir = os.path.join(os.path.dirname(__file__), "..", "debug")
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, "last_preprocessed_digit.png")
                debug_img = Image.fromarray(canvas_28.astype(np.uint8))
                debug_img.save(debug_path)
                print(f"Saved debug preprocessed image to: {debug_path}")
            except Exception as debug_err:
                print(f"Warning: Failed to save debug image: {debug_err}")

        # 11. Normalize pixel values from [0, 255] to [0.0, 1.0]
        normalized_array = canvas_28 / 255.0

        # 12. Expand batch dimension to (1, 28, 28)
        img_tensor = np.expand_dims(normalized_array, axis=0)
        return img_tensor

    def predict(self, image_bytes: bytes) -> dict:
        """
        Execute prediction on preprocessed image bytes and return predicted digit and confidence.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        # Preprocess input image bytes with MNIST-style pipeline
        img_tensor = self.preprocess_image(image_bytes)

        # Run inference
        predictions = self.model.predict(img_tensor, verbose=0)[0]

        # Extract class with highest softmax probability
        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        return {
            "digit": predicted_digit,
            "confidence": round(confidence, 4)
        }
