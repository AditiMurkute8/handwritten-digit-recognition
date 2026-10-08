import React, { useRef, useState } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import PredictionResult from './components/PredictionResult';

// Read API URL from Vite environment variable (VITE_API_URL) with default fallback
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
  const canvasRef = useRef(null);
  const [hasDrawn, setHasDrawn] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Clear canvas drawing and previous prediction state
  const handleClear = () => {
    if (canvasRef.current) {
      canvasRef.current.clear();
    }
    setResult(null);
    setError(null);
    setHasDrawn(false);
  };

  // Convert canvas to image Blob and send POST request to FastAPI backend
  const handlePredict = async () => {
    if (!canvasRef.current || canvasRef.current.isEmpty()) {
      setError('Please draw a digit on the canvas before clicking Predict.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Export canvas drawing as PNG Blob
      const imageBlob = await canvasRef.current.getBlob();
      if (!imageBlob) {
        throw new Error('Failed to capture canvas drawing.');
      }

      // Construct FormData payload with field name "file"
      const formData = new FormData();
      formData.append('file', imageBlob, 'digit.png');

      // Send request to FastAPI POST /predict endpoint
      // Note: Do NOT set Content-Type header manually so browser includes multipart boundary
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Prediction API Error:', err);
      setError(err.message || 'Failed to connect to prediction server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Handwritten Digit Recognition</h1>
        <p className="app-subtitle">Draw a digit (0–9) below to test the TensorFlow neural network model</p>
      </header>

      <main className="app-main">
        <div className="workspace-grid">
          <div className="canvas-section">
            <DrawingCanvas
              ref={canvasRef}
              onCanvasChange={(isDrawn) => {
                setHasDrawn(isDrawn);
                if (isDrawn && error) setError(null);
              }}
            />

            <div className="button-group">
              <button
                type="button"
                onClick={handleClear}
                className="btn btn-clear"
                disabled={loading}
              >
                Clear
              </button>

              <button
                type="button"
                onClick={handlePredict}
                className="btn btn-predict"
                disabled={loading || !hasDrawn}
              >
                {loading ? 'Predicting...' : 'Predict'}
              </button>
            </div>
          </div>

          <div className="result-section">
            <PredictionResult
              result={result}
              loading={loading}
              error={error}
            />
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Connected API: <code>{API_URL}</code></p>
      </footer>
    </div>
  );
}

export default App;
