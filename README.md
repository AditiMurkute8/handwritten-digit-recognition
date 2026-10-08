# Handwritten Digit Recognition

An end-to-end Machine Learning web application designed to recognize handwritten digits (0-9) using a convolutional neural network (CNN) trained on the MNIST dataset.

## Project Architecture & Directory Structure

```text
handwritten-digit-recognition/
├── backend/            # Python FastAPI server handling REST API requests and model inference
│   ├── app/            # Application logic, route handlers, and utility functions
│   ├── main.py         # Entry point for running the FastAPI application
│   └── requirements.txt# Python dependencies for the backend service
├── frontend/           # React single-page application for digit drawing & UI interaction
│   ├── public/         # Static web assets (HTML template, favicon, icons)
│   ├── src/            # React components, canvas drawing logic, and API calls
│   └── package.json    # Node.js dependencies and frontend scripts
├── model/              # Model training scripts, preprocessing tools, and saved model weights
│   ├── saved_models/   # Directory for exported model artifacts (.keras / .h5 files)
│   └── train.py        # TensorFlow/Keras training pipeline script
├── notebooks/          # Jupyter notebooks for data exploration, model experimentation, and evaluation
│   └── 01_mnist_exploration.ipynb # Initial EDA and prototype experiments
├── tests/              # Automated unit and integration test suite
│   ├── test_backend.py # API route and backend service tests
│   └── test_model.py   # Model inference and data pipeline tests
├── .gitignore          # Git exclusion rules for virtual environments, node_modules, and model binaries
└── README.md           # Project documentation and setup guide
```

## Directory Responsibilities

- **`backend/`**: Serves API endpoints to receive canvas image uploads from the frontend, preprocesses the image arrays, feeds them into the trained ML model, and returns digit prediction probabilities.
- **`frontend/`**: Interactive web interface featuring an HTML5 canvas component where users can draw digits, clear canvas, submit images to the backend, and view prediction results visually.
- **`model/`**: Contains Python scripts to download MNIST, train the CNN model, evaluate accuracy, and export saved model artifacts (`.keras`/`.h5`) for inference.
- **`notebooks/`**: Dedicated space for exploratory data analysis (EDA), data visualization, hyperparameter tuning, and prototyping model architectures before moving code into production modules.
- **`tests/`**: Isolated testing directory containing test suites for backend API endpoints, image preprocessing helpers, and model prediction logic.

## Planned Workflow
1. Train CNN model on MNIST dataset (`model/train.py`).
2. Serve trained model via FastAPI endpoints (`backend/main.py`).
3. Connect React drawing UI to backend API (`frontend/src`).
4. Containerize application components using Docker for reproducible deployment.
