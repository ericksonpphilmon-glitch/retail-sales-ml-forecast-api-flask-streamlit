# retail-sales-ml-forecast-api-flask-streamlit
Market Sales Forecasting — End-to-End ML Deployment
SuperKart Sales Forecasting — End-to-End ML Deployment
A full-stack machine learning solution that forecasts retail product sales revenue across store locations. Built with scikit-learn pipelines, served via a containerized Flask REST API, and surfaced through an interactive Streamlit UI — both deployed on Hugging Face Spaces.

Live Demo
ServiceLink🖥️ Frontend (Streamlit)Sales Estimate App⚙️ Backend (Flask API)Hugging Face Docker Space

What It Does
Given product and store attributes, the model predicts the total sales revenue (Product_Store_Sales_Total) for that product-store combination. The prediction endpoint is consumed by a Streamlit frontend that lets users interactively enter product details and receive a sales estimate in real time.
Input features:

Product weight, sugar content, allocated display area, MRP, and product type
Store establishment year, size, city tier, and store type

Output: Predicted sales revenue (continuous)

Architecture
User (Streamlit UI)
        │
        │  HTTP POST /predict
        ▼
Flask REST API  ←─ joblib model (Random Forest Pipeline)
        │
        │  JSON response
        ▼
Streamlit renders prediction
Both services run in Docker containers hosted on Hugging Face Spaces:

Backend: Python 3.9-slim + Gunicorn (4 workers, port 7860)
Frontend: Python 3.9-slim + Streamlit (port 8501)


Model
Two ensemble regressors were evaluated — XGBoost and Random Forest — each wrapped in a full scikit-learn Pipeline that handles preprocessing automatically at inference time.
Preprocessing pipeline:

StandardScaler on numeric features
OneHotEncoder (handle_unknown='ignore') on categorical features

Evaluation metric used for selection: MAPE (Mean Absolute Percentage Error), with R², MAE, MSE, and RMSE tracked as secondary metrics.
Hyperparameter tuning: GridSearchCV over depth, estimators, learning rate, regularization (XGB) and split/leaf/feature parameters (RF).
Selected model: Untuned Random Forest — achieved lower MAPE on the test set than the tuned XGBoost, indicating the XGB tuning introduced marginal overfitting relative to the RF baseline.
The final pipeline is serialized with joblib and loaded directly by the Flask app at startup.

Repository Structure
├── SuperKart_Forecasting.ipynb   # Full notebook: EDA → modeling → deployment
├── backend_files/
│   ├── app.py                    # Flask REST API
│   ├── model.joblib              # Serialized Random Forest pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── frontend_files/
│   ├── app.py                    # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
└── deployment_files/
    └── model.joblib              # Master serialized model

Running Locally
Backend:
bashcd backend_files
pip install -r requirements.txt
python app.py
# API available at http://localhost:7860
Frontend:
bashcd frontend_files
pip install -r requirements.txt
streamlit run app.py
# UI available at http://localhost:8501
Docker (backend):
bashcd backend_files
docker build -t superkart-api .
docker run -p 7860:7860 superkart-api
Docker (frontend):
bashcd frontend_files
docker build -t superkart-ui .
docker run -p 8501:8501 superkart-ui

Tech Stack
LayerTechnologyModelingscikit-learn, XGBoost, Random ForestPreprocessingscikit-learn Pipeline, StandardScaler, OneHotEncoderSerializationjoblibBackend APIFlask, GunicornFrontend UIStreamlitContainerizationDockerDeploymentHugging Face SpacesData wranglingpandas, numpyVisualizationmatplotlib, seaborn

Key Findings

Store type is the strongest sales predictor — Departmental Stores in Tier 1 cities generate significantly higher revenue than other store/location combinations, suggesting that location tier and store format should be primary considerations in expansion planning.
Product MRP correlates positively with sales revenue, making pricing strategy a meaningful lever alongside product placement.
Display area allocation shows diminishing returns beyond a moderate threshold — suggesting that floor space is better optimized across a broader product mix than concentrated on a single SKU.
Random Forest outperformed tuned XGBoost on out-of-sample MAPE, a useful reminder that ensemble baselines are competitive and hyperparameter tuning is not always additive.


Dependencies
numpy==2.0.2
pandas==2.2.2
scikit-learn==1.6.1
matplotlib==3.10.0
seaborn==0.13.2
joblib==1.4.2
xgboost==2.1.4
flask==2.2.2
gunicorn==20.1.0
streamlit==1.43.2
requests==2.28.1
huggingface_hub==0.34.0
