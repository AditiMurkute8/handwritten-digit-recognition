"""
Unit and integration tests for FastAPI backend and image preprocessing pipeline.
"""

import io
import pytest
import numpy as np
from PIL import Image, ImageDraw

from backend.app.model_service import ModelService


@pytest.fixture
def model_service():
    service = ModelService()
    return service


def create_sample_digit_image(digit_char="5", bg_white=True, size=(280, 280)):
    """Helper utility to generate raw canvas image bytes for unit testing."""
    bg_color = 255 if bg_white else 0
    fg_color = 0 if bg_white else 255
    img = Image.new("L", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    if digit_char == "5":
        # Draw digit 5 stroke
        draw.line([(50, 50), (200, 50)], fill=fg_color, width=20)
        draw.line([(50, 50), (50, 130)], fill=fg_color, width=20)
        draw.line([(50, 130), (200, 130)], fill=fg_color, width=20)
        draw.line([(200, 130), (200, 220)], fill=fg_color, width=20)
        draw.line([(200, 220), (50, 220)], fill=fg_color, width=20)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_output_shape(model_service):
    image_bytes = create_sample_digit_image("5")
    tensor = model_service.preprocess_image(image_bytes)
    assert tensor.shape == (1, 28, 28)
    assert tensor.dtype == np.float32


def test_preprocess_pixel_normalization(model_service):
    image_bytes = create_sample_digit_image("5")
    tensor = model_service.preprocess_image(image_bytes)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_white_background_inversion(model_service):
    # White background (255) with black stroke (0)
    image_bytes = create_sample_digit_image("5", bg_white=True)
    tensor = model_service.preprocess_image(image_bytes)[0]
    # Corners (background) should be 0.0 after inversion
    assert tensor[0, 0] == 0.0
    # Active stroke area should be > 0.0
    assert tensor.max() > 0.5


def test_black_background_support(model_service):
    # Black background (0) with white stroke (255)
    image_bytes = create_sample_digit_image("5", bg_white=False)
    tensor = model_service.preprocess_image(image_bytes)[0]
    assert tensor[0, 0] == 0.0
    assert tensor.max() > 0.5


def test_empty_canvas(model_service):
    img = Image.new("L", (280, 280), color=255)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    tensor = model_service.preprocess_image(buf.getvalue())
    assert tensor.shape == (1, 28, 28)
    assert np.all(tensor == 0.0)
