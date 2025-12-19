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
