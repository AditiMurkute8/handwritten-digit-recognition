import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from 'react';

/**
 * DrawingCanvas Component
 * Provides an HTML5 canvas for drawing handwritten digits (0-9)
 * Supports mouse, touch, and pointer events with smooth stroke drawing.
 */
const DrawingCanvas = forwardRef(({ onCanvasChange }, ref) => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [hasDrawn, setHasDrawn] = useState(false);
  const lastPointRef = useRef({ x: 0, y: 0 });

  // Initialize canvas with solid white background
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasDrawn(false);
    if (onCanvasChange) onCanvasChange(false);
  };

  useEffect(() => {
    clearCanvas();
  }, []);

  // Expose clear and getBlob helper methods to parent App via ref
  useImperativeHandle(ref, () => ({
    clear: clearCanvas,
    isEmpty: () => !hasDrawn,
    getBlob: () => {
      return new Promise((resolve) => {
        const canvas = canvasRef.current;
        if (!canvas) return resolve(null);
        canvas.toBlob((blob) => resolve(blob), 'image/png');
      });
    }
  }));

  // Calculate coordinates relative to canvas boundary
  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
    const clientY = e.clientY ?? (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
    return {
      x: clientX - rect.left,
      y: clientY - rect.top
    };
  };

  const startDrawing = (e) => {
    e.preventDefault();
    const point = getCoordinates(e);
    lastPointRef.current = point;
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault();

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const currentPoint = getCoordinates(e);

    ctx.beginPath();
    ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
    ctx.lineTo(currentPoint.x, currentPoint.y);

    // Thick black stroke suitable for 28x28 digit downsampling
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    lastPointRef.current = currentPoint;
    if (!hasDrawn) {
      setHasDrawn(true);
      if (onCanvasChange) onCanvasChange(true);
    }
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        onPointerDown={startDrawing}
        onPointerMove={draw}
        onPointerUp={stopDrawing}
        onPointerLeave={stopDrawing}
        className="drawing-canvas"
      />
    </div>
  );
});

export default DrawingCanvas;
