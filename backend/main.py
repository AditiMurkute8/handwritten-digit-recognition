"""
FastAPI Backend Application for Handwritten Digit Recognition API.

Provides endpoints for health check and digit image classification.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from app.model_service import ModelService
except ImportError:
    from backend.app.model_service import ModelService

# Initialize global ModelService instance
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Loads the trained Keras model into memory once on startup.
    """
    try:
        model_service.load_model()
    except Exception as e:
        print(f"Startup error: Failed to load model - {e}")
        raise e
    yield
    print("FastAPI application shutting down.")


# Create FastAPI application instance
app = FastAPI(
    title="Handwritten Digit Recognition API",
    description="FastAPI REST API serving TensorFlow/Keras MNIST model predictions.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware to enable requests from frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Pydantic Schema
class PredictionResponse(BaseModel):
    digit: int
    confidence: float


@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint returning service status and model readiness.
    """
    model_loaded = model_service.model is not None
    return {
        "status": "online",
        "message": "Handwritten Digit Recognition API is running.",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_digit(file: UploadFile = File(...)):
    """
    Accepts an uploaded digit image file, preprocesses it,
    runs inference using the trained Keras model, and returns predicted digit & confidence.
    """
    # 1. Validate file presence
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided."
        )

    # Validate file extension
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    filename = file.filename.lower() if file.filename else ""
    if filename and not filename.endswith(valid_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Allowed extensions: {', '.join(valid_extensions)}"
        )

    # 2. Read image bytes
    try:
        image_bytes = await file.read()
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty."
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading uploaded file: {str(e)}"
        )

    # 3. Preprocess and predict
    try:
        result = model_service.predict(image_bytes)
        return result
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {str(val_err)}"
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model inference failed: {str(err)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
