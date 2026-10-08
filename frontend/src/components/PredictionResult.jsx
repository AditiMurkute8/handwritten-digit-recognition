import React from 'react';

/**
 * PredictionResult Component
 * Displays the predicted digit, confidence score bar, loading state, or error message.
 */
const PredictionResult = ({ result, loading, error }) => {
  if (loading) {
    return (
      <div className="result-card loading-card">
        <div className="spinner"></div>
        <p>Analyzing handwritten digit...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-card error-card">
        <span className="error-icon">⚠️</span>
        <p className="error-message">{error}</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="result-card empty-card">
        <p className="placeholder-text">Draw a digit (0–9) on the canvas and click <strong>Predict</strong>.</p>
      </div>
    );
  }

  const confidencePercent = (result.confidence * 100).toFixed(1);

  return (
    <div className="result-card success-card">
      <h3 className="result-title">Prediction Result</h3>
      <div className="digit-display">{result.digit}</div>
      <div className="confidence-meter">
        <span className="confidence-label">Confidence Score:</span>
        <span className="confidence-value">{confidencePercent}%</span>
        <div className="confidence-bar-bg">
          <div
            className="confidence-bar-fill"
            style={{ width: `${confidencePercent}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;
