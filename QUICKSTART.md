# Quick Start Guide

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Model File

Ensure `Trained_model.keras` is in the project directory:

```bash
ls -lh Trained_model.keras
```

## Running the Server

### Option 1: Direct Run (Recommended for Development)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using Python

```bash
python main.py
```

### Option 3: Production Mode (Multiple Workers)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 4: Using Docker

```bash
# Build the image
docker build -t ml-api .

# Run the container
docker run -p 8000:8000 ml-api
```

### Option 5: Using Docker Compose

```bash
docker-compose up -d
```

## Testing the API

### 1. Check if Server is Running

```bash
curl http://localhost:8000/
```

### 2. Health Check

```bash
curl http://localhost:8000/health
```

### 3. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### 4. Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"features": [5.1, 3.5, 1.4, 0.2]},
    {"features": [6.2, 2.9, 4.3, 1.3]}
  ]'
```

### 5. Run Test Suite

```bash
python test_api.py
```

## Accessing Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Common Commands

### Check Server Logs

```bash
# If running with uvicorn
tail -f uvicorn.log

# If running with Docker
docker logs -f ml_prediction_api
```

### Stop the Server

```bash
# If running locally
Ctrl+C

# If running with Docker Compose
docker-compose down
```

### Restart the Server

```bash
# Docker Compose
docker-compose restart

# Docker
docker restart ml_prediction_api
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uvicorn main:app --port 8001
```

### Model Not Found

```bash
# Check if Trained_model.keras exists
ls -lh Trained_model.keras

# Check file permissions
chmod 644 Trained_model.keras
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Example Python Client

```python
import requests

# Create a session for connection pooling
session = requests.Session()

# Make a prediction
response = session.post(
    "http://localhost:8000/predict",
    json={"features": [5.1, 3.5, 1.4, 0.2]}
)

result = response.json()
print(f"Predicted Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}")
```

## Performance Tips

1. **Use multiple workers in production:**
   ```bash
   uvicorn main:app --workers 4
   ```

2. **Enable compression:**
   Add middleware in `main.py`

3. **Use connection pooling:**
   Clients should reuse HTTP connections

4. **Monitor with logs:**
   ```bash
   uvicorn main:app --log-level info --access-log
   ```

## Next Steps

1. ✅ Test the `/predict` endpoint with your actual data
2. ✅ Customize the `PredictionInput` schema for your model
3. ✅ Add authentication if needed
4. ✅ Set up monitoring and logging
5. ✅ Deploy to production (AWS, GCP, Azure, etc.)

## Support

- Check the main README.md for detailed documentation
- Visit http://localhost:8000/docs for interactive API documentation
- Run `python test_api.py` to verify everything works
