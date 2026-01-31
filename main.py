"""
FastAPI ML Model Serving Application - Music Genre Classification
This module provides a REST API for serving predictions from a trained Keras model.
Updated to support audio file uploads for music genre classification.
"""

import os
import io
import logging
from contextlib import asynccontextmanager
from pyexpat import features
from pyexpat import features
from typing import List, Dict, Any, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variable to store the loaded model
model = None
MODEL_PATH = "Trained_model.h5"

# Music genre labels (modify based on your model's output classes)
GENRE_LABELS = [
    "Blues", "Classical", "Country", "Disco", "Hip-Hop",
    "Jazz", "Metal", "Pop", "Reggae", "Rock",
    "Electronic", "Synthwave", "Indie", "R&B", "Folk"
]


# =============================================================================
# Audio Processing Functions
# =============================================================================

from pydub import AudioSegment


def extract_audio_features(audio_file: bytes) -> np.ndarray:
    try:
        # Decode audio safely using pydub (handles all WAV/MP3/OGG)
        audio = AudioSegment.from_file(io.BytesIO(audio_file))
        audio = audio.set_channels(1).set_frame_rate(22050)

        # Convert to numpy
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Normalize
        samples /= np.max(np.abs(samples))

        # Create Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=samples,
            sr=22050,
            n_mels=150,
            n_fft=2048,
            hop_length=512
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Ensure fixed size (150,150)
        if mel_db.shape[1] < 150:
            mel_db = np.pad(mel_db, ((0, 0), (0, 150 - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :150]

        # Normalize spectrogram
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        return mel_db.reshape(1, 150, 150, 1).astype(np.float32)

    except Exception as e:
        logger.warning(f"Rejected audio file (decode failed): {e}")
        return None




# =============================================================================
# Pydantic Models for Request/Response Validation
# =============================================================================

class PredictionInput(BaseModel):
    """
    Input schema for prediction requests (JSON-based).
    """
    features: List[float] = Field(
        ...,
        description="List of numerical features for prediction",
        example=[0.1, 0.2, 0.3, 0.4]
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate that features list is not empty"""
        if not v:
            raise ValueError("Features list cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("All features must be numerical values")
        return v


class MusicGenreResponse(BaseModel):
    """
    Output schema for music genre prediction responses.
    """
    status: str = Field(..., description="Request status")
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    analysis: Dict[str, Any] = Field(..., description="Genre analysis results")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "file_id": "audio_x922",
                "analysis": {
                    "primary_genre": "Synthwave",
                    "confidence": 0.942,
                    "all_predictions": {
                        "Synthwave": 0.942,
                        "Electronic": 0.887,
                        "Pop": 0.045
                    },
                    "bpm": 112,
                    "mood": "Cinematic"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    model_loaded: bool
    model_path: str


# =============================================================================
# Application Lifecycle Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles model loading at startup and cleanup at shutdown.
    """
    # Startup: Load the model
    global model
    logger.info("Starting application...")
    
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"Model loaded successfully")
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    logger.info("Shutting down application...")
    model = None


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Music Genre Classification API",
    description="REST API for music genre classification using deep learning",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Music Genre Classification API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST with audio file)",
            "predict_json": "/predict/json (POST with features)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service and model status.
    Returns the current status of the API and model.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=MODEL_PATH
    )


@app.post("/predict", response_model=MusicGenreResponse, tags=["Prediction"])
async def predict_audio_file(file: UploadFile = File(...)):
    """
    Music genre prediction endpoint for audio files.
    
    Accepts audio file uploads (mp3, wav, etc.) and returns genre predictions.
    
    Args:
        file: Uploaded audio file
        
    Returns:
        MusicGenreResponse containing prediction results
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Service unavailable."
        )
    
    try:
        # Validate file type
        allowed_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"Processing audio file: {file.filename}")
        
        # Read audio file
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Extract features from audio
        features = extract_audio_features(audio_bytes)

        # ✅ Polite rejection for broken files
        if features is None:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "rejected",
                    "file": file.filename,
                    "reason": "Unsupported or corrupted audio file"
                }
            )
        
        # Get expected input shape from model (excluding batch dimension)
        expected_shape = model.input_shape[1:]
        
        # Reshape input to match model's expected input shape
        if len(expected_shape) == 1:
            if len(features) != expected_shape[0]:
                # Pad or truncate features to match expected size
                if len(features) > expected_shape[0]:
                    features = features[:expected_shape[0]]
                else:
                    features = np.pad(features, (0, expected_shape[0] - len(features)))
            features_array = features.reshape(1, -1)
        else:
            features_array = features.reshape(1, *expected_shape)
        
        logger.info(f"Features shape: {features_array.shape}")
        
        # Make prediction
        prediction = model.predict(features_array, verbose=0)
        
        # Extract prediction results
        prediction_probs = prediction[0]
        predicted_class_idx = int(np.argmax(prediction_probs))
        confidence = float(np.max(prediction_probs))
        
        # Get genre label
        if predicted_class_idx < len(GENRE_LABELS):
            primary_genre = GENRE_LABELS[predicted_class_idx]
        else:
            primary_genre = f"Genre_{predicted_class_idx}"
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction_probs)[-3:][::-1]
        all_predictions = {}
        for idx in top_3_indices:
            if idx < len(GENRE_LABELS):
                genre_name = GENRE_LABELS[idx]
            else:
                genre_name = f"Genre_{idx}"
            all_predictions[genre_name] = float(prediction_probs[idx])
        
        # Generate unique file ID
        import hashlib
        file_id = f"audio_{hashlib.md5(audio_bytes[:1000]).hexdigest()[:8]}"
        
        logger.info(f"Prediction successful: {primary_genre} ({confidence:.4f})")
        
        # Return response in the format expected by frontend
        return MusicGenreResponse(
            status="success",
            file_id=file_id,
            analysis={
                "primary_genre": primary_genre,
                "confidence": round(confidence, 3),
                "all_predictions": all_predictions,
                "bpm": 112,  # Placeholder - implement BPM detection if needed
                "mood": "Energetic",  # Placeholder - implement mood detection if needed
                "instruments": ["Synthesizer", "Drum Machine"]  # Placeholder
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/json", tags=["Prediction"])
async def predict_json(input_data: PredictionInput):
    """
    Prediction endpoint for JSON-based feature input.
    
    This endpoint accepts pre-extracted features as JSON.
    Use this if you've already processed the audio file.
    
    Args:
        input_data: PredictionInput containing features
        
    Returns:
        Prediction results
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Service unavailable."
        )
    
    try:
        # Convert input features to numpy array
        features_array = np.array(input_data.features)
        
        # Get expected input shape from model (excluding batch dimension)
        expected_shape = model.input_shape[1:]
        
        # Reshape input to match model's expected input shape
        if len(expected_shape) == 1:
            if len(features_array) != expected_shape[0]:
                raise ValueError(
                    f"Expected {expected_shape[0]} features, got {len(features_array)}"
                )
            features_array = features_array.reshape(1, -1)
        else:
            features_array = features_array.reshape(1, *expected_shape)
        
        # Make prediction
        prediction = model.predict(features_array, verbose=0)
        
        # Extract prediction results
        prediction_list = prediction[0].tolist()
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        # Get genre label
        genre = GENRE_LABELS[predicted_class] if predicted_class < len(GENRE_LABELS) else f"Genre_{predicted_class}"
        
        logger.info(f"Prediction successful: {genre} ({confidence:.4f})")
        
        return {
            "prediction": prediction_list,
            "predicted_class": predicted_class,
            "predicted_genre": genre,
            "confidence": confidence
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for uncaught exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )


# =============================================================================
# Main Entry Point (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )
