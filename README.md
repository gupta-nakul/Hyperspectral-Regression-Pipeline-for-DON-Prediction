# Hyperspectral-Regression-Pipeline-for-DON-Prediction
A modular, scalable, and containerized machine learning pipeline for predicting DON concentration from hyperspectral data, featuring model selection, hyperparameter tuning, and API deployment

## Project Structure
hyperspectral_regression/
│── src/
│   ├── data/
│   │   ├── preprocess.py         # Data preprocessing functions, imputation, scaling, and optional PCA-based feature selection.
│   │   ├── dataset_loader.py     # Functions to load and (optionally) normalize the hyperspectral dataset.
│   │
│   ├── models/
│   │   ├── xgboost_model.py      # XGBoost regressor implementation.
│   │   ├── fcn_model.py          # Fully Connected Network (FCN) implementation using PyTorch.
│   │   ├── cnn_model.py          # 1D Convolutional Neural Network (CNN) implementation using PyTorch.
│   │   ├── transformer_model.py  # Experimental Transformer-based model for regression (debugging in progress).
│   │   ├── base_model.py         # Abstract base class to enforce a standard interface for all models.
│   │
│   ├── training/
│   │   ├── train.py              # Training pipeline, orchestrates training of all models and saving of best performers.
│   │   ├── optimizer.py          # (Not yet integrated) Hyperparameter optimization ideas using Optuna.
│   │
│   ├── evaluation/
│   │   ├── evaluate.py           # Evaluation metrics (MAE, RMSE, R²) and residual plotting functions.
│   │
│   ├── deployment/
│   │   ├── predictor.py          # Model inference; abstracts prediction across different model types.
│   │   ├── api.py                # FastAPI-based API to expose predictions.
│   │
│   ├── utils/
│   │   ├── logger.py             # Logger configuration and setup.
│   │   ├── helper.py             # Helper functions (e.g., choosing and saving the best model based on R² score).
│   │   ├── load_config.py        # Loads project configuration from config.yaml.
│
│── notebooks/
│   ├── exploratory_analysis.ipynb          # Initial data exploration.
│   ├── feature_selection_experiments.ipynb   # Experiments with feature selection techniques.
│   ├── model_benchmarking.ipynb              # Comparative analysis of the various models.
│
│── tests/
│   ├── test_data.py            # Unit tests for the data pipeline.
│   ├── test_models.py          # Unit tests for the machine learning models.
│   ├── test_api.py             # Tests for the API endpoints.
│
│── requirements.txt            # Project dependencies.
│── config.yaml                 # Global configuration settings for data, model parameters, preprocessing, training, and deployment.
│── docker-compose.yaml         # Docker Compose configuration for containerizing training and API services.
│── Dockerfile.train            # Dockerfile for the training pipeline.
│── Dockerfile.api              # Dockerfile for the API service.
│── run.py                      # Main entry point to execute the pipeline.
│── README.md                   # Overview and instructions on running the project.


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/gupta-nakul/Hyperspectral-Regression-Pipeline-for-DON-Prediction.git
   cd Hyperspectral-Regression-Pipeline-for-DON-Prediction
   ```


## Configuration
The project settings are defined in the config.yaml file. This file contains configuration options for data paths, model hyperparameters, preprocessing settings, training splits, and deployment model paths. Ensure you update any paths or parameters as needed before running the pipeline.


## Running the Pipeline

### Local Execution

1. **Data Loading, Training and Evaluation**
To run the entire pipeline (data loading, preprocessing, training, evaluation, and best model selection), execute:

```bash
python run.py
```

2. **API for Predictions**
The API is built using FastAPI and can be run using:

```bash
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
```

Then, you can access the API at http://localhost:8000. The API exposes endpoints for status checking (GET /) and predictions (POST /predict).

### Docker-based Execution
The project includes Dockerfiles for containerizing the training and API services.

**Using Docker Compose:**

A docker-compose.yaml file is provided to run both training and API services together. From the project root, run:

```bash
docker-compose --build
```

This command will:
1. Build the training container using Dockerfile.train and run python run.py.
2. Build the API container using Dockerfile.api and expose the API on port 8000.
3. Share a Docker volume (models_data) between the services to ensure the API uses the best model produced during training.

Now to run each container, run:
1. **For training container:**
```bash
docker-compose run train
```

2. **For API container:**
```bash
docker-compose up api
```