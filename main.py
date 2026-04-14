from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# --- APP INITIALIZATION ---
app = FastAPI(title="PSL Translator API")

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION & MODEL LOADING ---
# Load actions from file if exists, otherwise detect from folders
if os.path.exists('actions.npy'):
    actions = np.load('actions.npy')
    print(f"Loaded {len(actions)} actions from actions.npy")
else:
    DATA_PATH = os.path.join('MP_Data')
    actions = np.array(sorted([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]))
    print(f"Detected {len(actions)} actions from folders")
model_path = 'psl_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = None

# --- DATA MODELS ---
class LandmarkData(BaseModel):
    # This should be a list of 30 frames, each containing 126 landmark values (2 hands * 21 points * 3)
    landmarks: list[list[float]] 

# --- ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(data: LandmarkData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    try:
        input_data = np.array(data.landmarks)
        
        # Validate shape (30, 126)
        if input_data.shape != (30, 126):
            # Attempt to flatten if passed as (30, 2, 21, 3) or similar
            if input_data.size == 30 * 126:
                input_data = input_data.reshape(30, 126)
            else:
                raise ValueError(f"Invalid input shape: {input_data.shape}. Expected (30, 126)")
        
        input_data = np.expand_dims(input_data, axis=0)
        # Optimized prediction: using model() instead of model.predict()
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        prediction = model(input_tensor, training=False).numpy()[0]
        predicted_index = np.argmax(prediction)
        
        return {
            "prediction": actions[predicted_index],
            "confidence": float(prediction[predicted_index]),
            "all_predictions": {actions[i]: float(prediction[i]) for i in range(len(actions))}
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("      PSL TRANSLATOR BACKEND IS STARTING")
    print("="*50)
    print(f"Loaded {len(actions)} actions from dataset.")
    if model:
        print("Model status: LOADED SUCCESSFULLY")
    else:
        print("Model status: WARNING - MODEL NOT FOUND (psl_model.h5)")
    
    print("\nIMPORTANT: Keep this window open while using the web app.")
    print(f"API Endpoint: http://localhost:8000")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
