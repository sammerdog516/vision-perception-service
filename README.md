# vision-perception-service
Problem

Modern robotics and AI systems rely on vision-based perception pipelines that must handle noisy sensor input, run ML inference reliably, and expose results to downstream systems in real time.
Many student projects stop at training a model and never address production deployment, latency, or system integration.

This project implements a robotics-style vision perception service: from raw image input, through sensor noise and preprocessing, to neural network inference, decision logic, and a deployed backend API.

Architecture
[ Image Input ]
   (upload / camera)
        ↓
[ Sensor Noise + Preprocessing ]
        ↓
[ ML Inference Layer ]
   ├── Neural Network (from scratch)
   └── CNN (PyTorch)
        ↓
[ Decision / Control Stub ]
        ↓
[ FastAPI Backend ]
        ↓
[ Deployed API + Metrics ]

Tech Stack

Languages

Python 3.11

Machine Learning

NumPy

PyTorch

OpenCV

Backend

FastAPI

Pydantic

Uvicorn

Infrastructure

Docker

Railway / Fly.io (deployment)

Data

MNIST (initial prototype)

MVP Scope (Locked)
Included

Neural network implemented from scratch

CNN baseline for comparison

Image preprocessing + noise simulation

Model inference API

Simple decision/control logic stub

Inference latency metrics

Dockerized deployment

Publicly accessible API

Explicitly Excluded (No Feature Creep)

No frontend UI

No training pipeline in production

No real robot hardware

No auth system beyond basic API key (optional)

No additional datasets beyond MNIST

Goal

Demonstrate end-to-end perception system engineering for AI and robotics applications:

ML fundamentals

Production backend

Deployment

Systems thinking

Folder Structure & Purpose

app/main.py
Orchestrates the application: initializes the FastAPI app, defines API routes, and connects preprocessing, inference, and control logic.

app/models/
Defines and loads machine learning models, including the from-scratch neural network and the CNN.

app/preprocessing/
Handles input preparation and sensor simulation, including image normalization, resizing, and noise injection.

app/inference/
Runs model inference, selects the appropriate model, and measures inference latency.

app/control/
Implements decision logic based on model outputs, serving as a stub for robotics-style perception-to-action pipelines.

Logging Strategy

The system logs operational signals for debugging and performance monitoring.

Log the model used for inference (cnn or nn)

Log inference latency (milliseconds)

Log input validation errors and inference failures

Logs are written to standard output for visibility in local runs and deployed environments.

Error Categories

Input Validation Errors (Client Errors)
Invalid or malformed inputs such as non-image files, incorrect shapes, or unsupported formats. Returned as HTTP 400 or 422.

Model / Inference Errors (System Errors)
Failures during model loading or inference, such as missing weights or runtime errors. Returned as HTTP 500 and logged internally.

Unexpected Exceptions (Fail-Safe Errors)
Unhandled edge cases or bugs. The service returns HTTP 500 while logging diagnostic information without crashing.

## Baseline Model
A fully connected neural network implemented from scratch to establish a learning baseline.  
Achieves ~97% accuracy on MNIST and highlights the limitations of non-convolutional architectures for vision tasks.

## Future Extensions (Robotics-Oriented)
- TODO: continuous image stream ingestion
- TODO: temporal smoothing across frames
- TODO: decision thresholds based on confidence
- TODO: integrate with real camera hardware
