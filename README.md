# 🤟 Real-Time Sign Language Translator

![GitHub License](https://img.shields.io/github/license/mdnadeemm/sign-language-translator)
![GitHub Repo Size](https://img.shields.io/github/repo-size/mdnadeemm/sign-language-translator)

An end-to-end Machine Learning application that translates American Sign Language (ASL) into text in real-time. This project features a React-based frontend for capturing webcam feeds and extracting hand landmarks, communicating via WebSockets with a high-performance FastAPI backend that runs a trained Scikit-Learn classifier to predict ASL gestures.

---

## 🧠 How It Works (System Architecture)

The system is designed for ultra-low latency real-time translation. The workflow is divided between the client (browser) and the server:

### 1. Client-Side Hand Tracking (React + MediaPipe)
- The React application accesses the user's webcam using `navigator.mediaDevices`.
- Each video frame is processed directly in the browser using **Google's MediaPipe Hand Tracking API**.
- MediaPipe extracts **21 3D landmarks** (x, y, z coordinates) for a detected hand.
- Performing landmark extraction on the client-side drastically reduces the network bandwidth required, as only a small JSON payload of numeric coordinates is sent over the network instead of raw video frames.

### 2. Real-Time Communication (WebSockets)
- The extracted 21 landmark coordinates are serialized into a JSON array.
- This payload is transmitted to the backend server over a persistent **WebSocket connection**, ensuring minimal overhead and latency compared to standard HTTP polling.

### 3. Server-Side Classification (FastAPI + Scikit-Learn)
- The **FastAPI** WebSocket endpoint receives the landmark array.
- The raw (x, y, z) coordinates are normalized to be invariant to the hand's position and distance from the camera (scale and translation invariance).
- The normalized features are fed into a pre-trained **Scikit-Learn Classifier** (e.g., Random Forest or SVM).
- The model predicts the corresponding ASL character or word.
- The predicted text, along with a confidence score, is instantly sent back through the WebSocket to the React frontend, where it is displayed to the user.

---

## 📁 Repository Structure

```text
sign-language-translator/
│
├── frontend/    # 🖥️ The User Interface (React)
│                # Captures your webcam video and extracts hand movements.
│
├── backend/     # ⚙️ The Server (FastAPI)
│                # Receives hand data in real-time and runs the prediction model.
│
└── ml/          # 🧠 The Machine Learning Pipeline (Python)
                 # Scripts to collect your own data and train the ASL model.
```

---

## 🔬 Data Collection & Model Training

If you want to train your own custom signs or improve the existing model, you can use the built-in ML pipeline located in the `ml/` directory.

### 1. Collecting Data
Data collection is done via a Python script that opens your webcam using OpenCV and extracts landmarks using MediaPipe.
- Run `python ml/collect_data.py`.
- You will be prompted to enter the label (e.g., "A", "B", "Hello") for the sign you are about to perform.
- The script records multiple frames of your hand, extracts the 21 landmarks, flattens the (x, y, z) coordinates, and appends them as a row to `ml/landmarks.csv` along with the label.

### 2. Training the Model
Once you have collected enough data points in `landmarks.csv`:
- Run `python ml/train_model.py`.
- This script reads the CSV, splits the data into training and testing sets, and trains a **Scikit-Learn Classifier** (like a Random Forest or Support Vector Machine).
- After training, it evaluates the accuracy and serializes the best model into `ml/model.pkl`.
- *Note: Ensure you copy your updated `model.pkl` to the appropriate location used by your `backend` so the FastAPI server loads the new weights.*

---

## 🚀 Getting Started

Follow these step-by-step instructions to run the application locally.

### Prerequisites
- **Node.js** (v18.x or later)
- **Python** (3.10 or later)
- A working **Webcam**

### 1. Clone the Repository

```bash
git clone https://github.com/mdnadeemm/sign-language-translator.git
cd sign-language-translator
```

### 2. Backend Server Setup

The backend handles the machine learning inference via FastAPI.

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server (runs on http://localhost:8000)
uvicorn app.main:app --reload
```

### 3. Frontend Client Setup

Open a new terminal window/tab to start the React UI.

```bash
cd frontend

# Install Node modules
npm install

# Start the Vite development server (runs on http://localhost:5173)
npm run dev
```

### 4. Using the App
1. Open your browser and navigate to `http://localhost:5173`.
2. Your browser will ask for **Camera Permissions**. Click **Allow**.
3. Position your hand in front of the camera. The MediaPipe overlay will draw connections between your joints.
4. Perform an ASL sign. The translated character will appear instantly on the screen!

---

## 🛠️ Tech Stack Specifics

- **Frontend Framework:** React 18 with Vite for blazing-fast HMR.
- **Computer Vision (Client):** `@mediapipe/hands` and `@mediapipe/camera_utils` for in-browser 60fps tracking.
- **Backend Framework:** FastAPI with Uvicorn (ASGI) for asynchronous WebSocket handling.
- **Machine Learning:** `scikit-learn` for training the classifier, `pandas` & `numpy` for data manipulation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
