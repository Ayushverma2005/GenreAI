# ML Model Prediction API - FastAPI + Keras

A production-ready REST API for serving predictions from a trained TensorFlow/Keras model.

## Features

- ✅ FastAPI framework for high performance
- ✅ Model loaded once at startup (not per request)
- ✅ Input validation using Pydantic
- ✅ Comprehensive error handling
- ✅ Batch prediction support
- ✅ Health check endpoint
- ✅ Auto-generated OpenAPI documentation
- ✅ Logging and monitoring
- ✅ Production-ready code structure

## Requirements

```bash
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
tensorflow>=2.14.0
numpy>=1.24.0
pydantic>=2.0.0
```

## Installation

1. **Install dependencies:**

```bash
pip install fastapi uvicorn tensorflow numpy pydantic
```

2. **Place your model file:**

Ensure your `Trained_model.keras` file is in the same directory as `main.py`

## Running the Server

### Development Mode (with auto-reload)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Python directly

```bash
python main.py
```

The server will start at: `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL:** `GET /`
- **Description:** API information

### 2. Health Check
- **URL:** `GET /health`
- **Description:** Check if service and model are ready

### 3. Single Prediction
- **URL:** `POST /predict`
- **Description:** Get prediction for a single input

### 4. Batch Prediction
- **URL:** `POST /predict/batch`
- **Description:** Get predictions for multiple inputs

### 5. Interactive Documentation
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Example Requests

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "Trained_model.keras"
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

**Response:**
```json
{
  "prediction": [0.9, 0.05, 0.05],
  "predicted_class": 0,
  "confidence": 0.9
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"features": [5.1, 3.5, 1.4, 0.2]},
    {"features": [6.2, 2.9, 4.3, 1.3]},
    {"features": [7.3, 2.9, 6.3, 1.8]}
  ]'
```

**Response:**
```json
[
  {
    "prediction": [0.9, 0.05, 0.05],
    "predicted_class": 0,
    "confidence": 0.9
  },
  {
    "prediction": [0.1, 0.8, 0.1],
    "predicted_class": 1,
    "confidence": 0.8
  },
  {
    "prediction": [0.05, 0.15, 0.8],
    "predicted_class": 2,
    "confidence": 0.8
  }
]
```

### Using Python Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Single prediction
payload = {
    "features": [5.1, 3.5, 1.4, 0.2]
}
response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())

# Batch prediction
batch_payload = [
    {"features": [5.1, 3.5, 1.4, 0.2]},
    {"features": [6.2, 2.9, 4.3, 1.3]}
]
response = requests.post("http://localhost:8000/predict/batch", json=batch_payload)
print(response.json())
```

## Error Handling

The API provides detailed error messages for common issues:

### Invalid Input (422)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": []}'
```

**Response:**
```json
{
  "detail": [
    {
      "loc": ["body", "features"],
      "msg": "Features list cannot be empty",
      "type": "value_error"
    }
  ]
}
```

### Model Not Loaded (503)

```json
{
  "detail": "Model is not loaded. Service unavailable."
}
```

### Prediction Error (500)

```json
{
  "detail": "Prediction failed: <error details>"
}
```

## Customization

### Modify Input Schema

Edit the `PredictionInput` class in `main.py`:

```python
class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="Your custom description",
        example=[1.0, 2.0, 3.0, 4.0]  # Your example
    )
    
    # Add custom validation
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 10:  # Example: enforce specific length
            raise ValueError("Expected exactly 10 features")
        return v
```

### Change Model Path

Modify the `MODEL_PATH` variable:

```python
MODEL_PATH = "/path/to/your/Trained_model.keras"
```

### Add Additional Endpoints

Add new endpoints in the main FastAPI app:

```python
@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "total_params": model.count_params()
    }
```

## Production Deployment

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY Trained_model.keras .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```

### Using Gunicorn + Uvicorn Workers

```bash
pip install gunicorn

gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Environment Variables

Add environment-based configuration:

```python
import os

MODEL_PATH = os.getenv("MODEL_PATH", "Trained_model.keras")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
```

## Monitoring and Logging

Logs are written to stdout with the format:
```
2024-01-30 10:30:45 - main - INFO - Loading model from Trained_model.keras
2024-01-30 10:30:46 - main - INFO - Model loaded successfully
```

For production, consider integrating:
- **Prometheus** for metrics
- **Sentry** for error tracking
- **ELK Stack** for log aggregation

## Performance Optimization

1. **Use multiple workers:**
   ```bash
   uvicorn main:app --workers 4
   ```

2. **Enable response compression:**
   ```python
   from fastapi.middleware.gzip import GZipMiddleware
   app.add_middleware(GZipMiddleware, minimum_size=1000)
   ```

3. **Add caching for frequent predictions:**
   ```python
   from functools import lru_cache
   ```

4. **Use async preprocessing if needed**

## Troubleshooting

### Model file not found
- Ensure `Trained_model.keras` is in the same directory as `main.py`
- Check file permissions
- Verify the MODEL_PATH variable

### Import errors
- Install all required dependencies
- Use a virtual environment
- Check TensorFlow compatibility with your system

### Port already in use
- Change the port: `uvicorn main:app --port 8001`
- Kill the process using the port

## License

MIT License

## Support

For issues and questions, please check:
- FastAPI documentation: https://fastapi.tiangolo.com/
- TensorFlow documentation: https://www.tensorflow.org/
