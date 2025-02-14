# Customer Complaints Classification Pipeline

An end-to-end pipeline that classifies customer complaints into specific categories using **machine learning** and **NLP** techniques. This project leverages **Prefect** for orchestration, **MLflow** for experiment tracking and model registry, **FastAPI** for serving the model as a microservice, and **Docker** for containerization.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Data Flow](#data-flow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Serving](#model-serving)
- [Docker Deployment](#docker-deployment)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This pipeline aims to **automatically classify customer complaints** into predefined categories. By applying text preprocessing (cleaning, tokenization, lemmatization) and training multiple models (e.g., Logistic Regression, SVM), we can predict which category a given complaint belongs to. The final (Champion) model is exposed via a **FastAPI** microservice, ready to serve real-time predictions.

Key objectives:

- **Automate** data loading, cleaning, and model training with **Prefect**.
- **Track** and **compare** multiple experiments using **MLflow**.
- **Deploy** a robust **REST API** with **FastAPI**, containerized for easy scaling.

---

## Features

1. **Data Preprocessing**  
   - Expands contractions, cleans text, tokenizes, removes stopwords, lemmatizes, etc.

2. **Training & Hyperparameter Tuning**  
   - Implements multiple classification models (Logistic Regression, SVM).
   - Uses GridSearchCV to find optimal hyperparameters.

3. **Experiment Tracking**  
   - Logs parameters, metrics, and artifacts (LabelEncoder, TF-IDF Vectorizer) in MLflow.
   - Registers models in MLflow Model Registry, identifying Champion/Challenger.

4. **Orchestration with Prefect**  
   - Defines tasks like `load_and_prepare_data()`, `split_data()`, and `train_and_evaluate_models()` as Prefect tasks.
   - Creates a flow to sequence these tasks.

5. **REST API for Inference**  
   - **FastAPI** microservice that loads the Champion model from MLflow, preprocesses input text, and returns predictions.

6. **Dockerized**  
   - Dockerfiles for the model API and UI for consistent and scalable deployment.

---

## Tech Stack

- **Languages**: Python 3.9
- **Libraries**: scikit-learn, pandas, nltk, Prefect, MLflow, FastAPI
- **Container**: Docker and docker-compose
- **Tracking**: DagsHub + MLflow for experiment logging and artifact storage

---

## Data Flow

1. **Raw Data**: (JSON/csv) containing raw complaint texts.
2. **Prefect Flow**: 
   - Loads data.
   - Cleans and preprocesses it.
   - Splits it into training and test sets.
   - Trains multiple models (LogReg, SVM), logs results to MLflow.
3. **Champion Model**: 
   - Saved in MLflow Model Registry.
   - Loaded by the FastAPI service at runtime.
4. **Inference**:
   - User sends a `POST` request to `/predict` with complaint text.
   - FastAPI preprocesses the text, vectorizes it, and uses the loaded model for classification.

---

## Project Structure

```bash
.
├── app
│   ├── model
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── pruebas.ipynb
│   │   └── requirements.txt
│   ├── UI
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   └── docker-compose.yaml
├── data
│   ├── clean_data
│   │   └── clean_tickets.csv
│   ├── processed_data
│   │   └── processed_tickets.csv
│   └── raw_data
│       └── tickets_classification_eng.json
├── experiments
│   └── model_experiments.ipynb
├── notebooks
│   ├── Data_Wrangling.ipynb
│   └── EDA.ipynb
├── training_pipeline
│   ├── models
│   │   ├── label_encoder.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── training_pipeline.py
├── .gitignore
└── README.md