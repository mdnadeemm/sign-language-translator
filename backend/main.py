from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import json
import os

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'model.pkl')

model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Warning: Model not found. Please train the model first.")

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            landmarks = json.loads(data)
            
            if model and len(landmarks) > 0:
                # landmarks is a list of 21 (x, y, z) dictionaries or a flat array
                # Let's assume the frontend sends a flat array of 63 floats
                features = np.array(landmarks).reshape(1, -1)
                
                # Predict
                prediction = model.predict(features)[0]
                
                # Send back the prediction
                await websocket.send_json({"prediction": str(prediction)})
            else:
                await websocket.send_json({"prediction": "Waiting for model/data"})
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
