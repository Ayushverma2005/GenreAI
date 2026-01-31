# Frontend-Backend Integration Guide
## Music Genre AI - Prediction Interface

---

## 🎯 Quick Overview

The `ai_genre_prediction.html` file is now fully integrated with your FastAPI backend. Users can upload audio files, and the AI will predict the music genre in real-time.

---

## 🔌 How It Works

### **Request Flow:**

```
User Selects Audio File
        ↓
JavaScript Validates File (type, size)
        ↓
User Clicks "Analyze Genre Now"
        ↓
FormData Created with Audio File
        ↓
POST Request to http://127.0.0.1:8000/predict
        ↓
FastAPI Backend Processes Audio
        ↓
Model Predicts Genre + Confidence
        ↓
JSON Response Returned
        ↓
Frontend Displays Results (no page reload)
```

---

## 📡 API Communication

### **Frontend Request (JavaScript):**

```javascript
// Create FormData with audio file
const formData = new FormData();
formData.append('file', selectedFile);

// Send to backend
const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    body: formData
});

// Parse JSON response
const result = await response.json();
```

### **Expected Backend Response Format:**

The frontend supports **two response formats**:

**Format 1 (Current Backend):**
```json
{
  "status": "success",
  "file_id": "audio_x922",
  "analysis": {
    "primary_genre": "Rock",
    "confidence": 0.876
  }
}
```

**Format 2 (Simple):**
```json
{
  "predicted_genre": "Rock",
  "confidence": 0.876
}
```

The frontend will automatically detect and handle both formats.

---

## ✅ What Was Integrated

### **Frontend Changes:**

1. ✅ **File Upload Zone** - Click or drag-drop to select audio files
2. ✅ **File Validation** - Checks for valid audio formats (MP3, WAV, OGG, FLAC, M4A) and size (max 50MB)
3. ✅ **Analyze Button** - Disabled until file selected, sends request to backend
4. ✅ **Loading State** - Shows spinner while processing
5. ✅ **Results Display** - Updates genre and confidence with smooth animations
6. ✅ **Error Handling** - Shows user-friendly error messages
7. ✅ **Status Updates** - "Ready for Upload" → "Processing Audio" → "Analysis Complete"

### **Key JavaScript Functions:**

- `handleFileSelect(file)` - Validates and stores selected file
- `analyzeBtn.addEventListener('click')` - Sends file to backend via fetch()
- `displayPrediction(data)` - Updates UI with prediction results
- `showError(message)` - Displays error messages

---

## 🚀 Running the Application

### **Step 1: Start the Backend**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Verify it's running:**
```bash
curl http://127.0.0.1:8000/health
```

### **Step 2: Open the Frontend**

**Option A - Direct File Access:**
```bash
open ai_genre_prediction.html
```

**Option B - Local Server (Recommended to avoid CORS issues):**
```bash
python -m http.server 8080
# Then visit: http://localhost:8080/ai_genre_prediction.html
```

### **Step 3: Test the Integration**

1. Click on the upload zone or drag an audio file
2. Verify the filename appears below the upload zone
3. Click "Analyze Genre Now"
4. Watch the loading spinner
5. See the prediction results appear with genre and confidence

---

## 🔍 Backend Requirements

Your FastAPI backend **must have**:

### **1. CORS Middleware (Already in main.py)**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### **2. /predict Endpoint**

```python
@app.post("/predict")
async def predict_audio_file(file: UploadFile = File(...)):
    # Process audio file
    # Extract features
    # Run model prediction
    # Return JSON response
    return {
        "status": "success",
        "file_id": "audio_123",
        "analysis": {
            "primary_genre": "Rock",
            "confidence": 0.876
        }
    }
```

**No changes needed** - your existing backend already supports this!

---

## 🐛 Troubleshooting

### **Issue: CORS Error**

**Error in Console:**
```
Access to fetch at 'http://127.0.0.1:8000/predict' from origin 'null' 
has been blocked by CORS policy
```

**Solution:**
- Use `python -m http.server` to serve HTML (not file://)
- Backend already has CORS middleware enabled

---

### **Issue: Connection Refused**

**Error in Console:**
```
Failed to fetch: TypeError: Failed to fetch
```

**Solutions:**
1. Check backend is running: `curl http://127.0.0.1:8000/health`
2. Verify correct port (8000)
3. Check firewall settings

---

### **Issue: Invalid File Type**

**Error in UI:**
```
Invalid file type. Please upload an audio file (MP3, WAV, OGG, FLAC, M4A)
```

**Solution:**
- Frontend validates before upload
- Backend also validates on server
- Only use supported audio formats

---

### **Issue: File Too Large**

**Error in UI:**
```
File too large. Maximum size is 50MB
```

**Solution:**
- Frontend checks file size before upload
- Compress audio file or use shorter clip
- Adjust max size in code if needed

---

## 🎨 UI Features

### **Visual Feedback:**

1. **Upload Zone:**
   - Pulses with purple glow
   - Changes on hover
   - Shows selected filename

2. **Analyze Button:**
   - Disabled when no file selected
   - Shows "Select Audio File First"
   - Changes to "Analyze Genre Now" when ready

3. **Status Indicator:**
   - "Ready for Upload" (idle)
   - "Processing Audio..." (during API call)
   - "Analysis Complete" (success)
   - "Analysis Failed" (error)

4. **Results Display:**
   - Animated confidence circle (0-100%)
   - Genre name in bold uppercase
   - Smooth fade-in animation
   - Sound wave visualization continues

5. **Error Alerts:**
   - Red-themed alert box
   - Clear error icon
   - Descriptive error message

---

## 🔧 Customization

### **Change API URL:**

Line 327:
```javascript
const API_URL = 'http://127.0.0.1:8000/predict';
// For production:
// const API_URL = 'https://api.yourapp.com/predict';
```

### **Adjust File Size Limit:**

Line 419:
```javascript
const maxSize = 50 * 1024 * 1024;  // 50MB
// Change to 100MB:
// const maxSize = 100 * 1024 * 1024;
```

### **Add More File Types:**

Line 409:
```javascript
const validExtensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a'];
// Add more:
// const validExtensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'];
```

---

## 📊 Response Handling

The frontend automatically handles multiple response formats:

```javascript
// Checks for integrated backend format
if (data.analysis && data.analysis.primary_genre) {
    genre = data.analysis.primary_genre;
    confidence = data.analysis.confidence;
}
// Falls back to simple format
else if (data.predicted_genre) {
    genre = data.predicted_genre;
    confidence = data.confidence;
}
```

This means your backend can return **either format** and it will work!

---

## 🎯 Key Points

✅ **No Backend Changes Required** - Your existing FastAPI backend works as-is

✅ **No Page Reload** - Results display dynamically using JavaScript

✅ **Drag & Drop Support** - Users can drag files or click to browse

✅ **Full Error Handling** - Validates files client-side before upload

✅ **Beautiful Animations** - Confidence circle animates smoothly

✅ **Mobile Friendly** - Responsive design works on all devices

---

## 📝 Summary

**What happens when user uploads a file:**

1. User selects/drops audio file
2. Frontend validates file type and size
3. User clicks "Analyze Genre Now"
4. JavaScript creates FormData and sends POST to `/predict`
5. FastAPI backend processes audio and returns JSON
6. Frontend displays genre and confidence with animations
7. No page reload - all happens seamlessly!

**That's it! Your Music Genre AI is fully integrated and ready to use! 🎵🎸**
