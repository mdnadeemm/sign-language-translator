import { useEffect, useRef, useState } from 'react'
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import './App.css'

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const handLandmarkerRef = useRef(null)
  
  const [prediction, setPrediction] = useState('Waiting for model...')
  const [isModelReady, setIsModelReady] = useState(false)
  const [isConnected, setIsConnected] = useState(false)

  // Initialize WebSockets
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/predict')
    ws.onopen = () => setIsConnected(true)
    ws.onclose = () => setIsConnected(false)
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setPrediction(data.prediction)
    }
    wsRef.current = ws

    return () => ws.close()
  }, [])

  // Initialize MediaPipe and Webcam
  useEffect(() => {
    async function initMediaPipe() {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      )
      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
      })
      handLandmarkerRef.current = handLandmarker
      setIsModelReady(true)
      startWebcam()
    }
    
    initMediaPipe()
  }, [])

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error("Error accessing webcam:", err)
    }
  }

  const handleVideoLoaded = () => {
    predictWebcam()
  }

  const predictWebcam = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !handLandmarkerRef.current) return

    const startTimeMs = performance.now()
    const results = handLandmarkerRef.current.detectForVideo(video, startTimeMs)

    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (results.landmarks && results.landmarks.length > 0) {
      // Draw landmarks
      for (const landmarks of results.landmarks) {
        for (const landmark of landmarks) {
          ctx.beginPath()
          ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI)
          ctx.fillStyle = "#00FF00"
          ctx.fill()
        }
      }

      // Send to backend if WebSocket is open
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Flatten the array: [x1, y1, z1, x2, y2, z2...]
        const flatLandmarks = results.landmarks[0].flatMap(l => [l.x, l.y, l.z])
        wsRef.current.send(JSON.stringify(flatLandmarks))
      }
    } else {
      // If no hands detected, you could send empty array to backend to clear prediction
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        setPrediction('No hand detected')
      }
    }

    // Call predictWebcam again
    requestAnimationFrame(predictWebcam)
  }

  return (
    <div className="container">
      <header className="header">
        <h1>ASL Translator</h1>
        <div className={`status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '● Connected to Server' : '○ Disconnected'}
        </div>
      </header>
      
      <main className="main-content">
        <div className="video-wrapper">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline 
            onLoadedData={handleVideoLoaded}
            style={{ width: '640px', height: '480px' }}
          ></video>
          <canvas 
            ref={canvasRef}
            width="640"
            height="480"
            className="overlay-canvas"
          ></canvas>
          {!isModelReady && <div className="loading-overlay">Loading Models...</div>}
        </div>

        <div className="prediction-box">
          <div className="prediction-label">Translation</div>
          <div className="prediction-text">{prediction}</div>
        </div>
      </main>
    </div>
  )
}

export default App
