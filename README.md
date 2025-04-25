<p align="center">
  <img src="frontend/images/Cocolytics.png" alt="Cocolytics AI Logo" width="200" />
</p>

# Cocolytics AI  
### Coconut-Price-Prediction-using-TFT

**An end-to-end application for forecasting Sri Lankan coconut prices using the Temporal Fusion Transformer (TFT)**

---

## 🚀 Overview

**Cocolytics AI** is the **first multivariate, attention-based forecasting system** built specifically for Sri Lanka’s coconut market. Leveraging 28 years of monthly retail prices and key exogenous drivers (district-level producer prices, diesel costs, USD/LKR exchange rates), it delivers:

- **State-of-the-art accuracy** via the Temporal Fusion Transformer  
- **Interpretability** through dynamic attention heat-maps  
- **Regional fairness** across major coconut-growing districts  
- **Reproducible, open-source pipeline** from data ingestion to deployment  

This repository implements the full research pipeline behind our paper “Cocolytics AI: A Multivariate Approach for Forecasting Coconut Prices in Sri Lanka.”

---

## 📦 Features

- **Data Ingestion & Cleaning**  
  – Central Bank, Coconut Development Authority, Ceypetco & Investing.com sources  
  – Automated alignment, interpolation & outlier winsorization  
- **Feature Engineering**  
  – 12-month lags for classical baselines  
  – 60+12 encoder/decoder sequences for TFT  
- **Model Zoo**  
  – Classical: ARIMA, SARIMA, ARIMAX/SARIMAX, VAR  
  – ML: XGBoost  
  – Deep Learning: Temporal Fusion Transformer  
- **Evaluation**  
  – Fixed hold-out & 60-fold rolling-origin CV  
  – MAE, RMSE, MAPE, sMAPE & regional fairness metrics  
- **Interpretability**  
  – Attention-based driver attributions over time  
- **Deployment**  
  – FastAPI service & interactive React/Tailwind dashboard  

---

## 📖 Getting Started

### Prerequisites

- **Python 3.8+**  
- **CUDA 11.8+** (for GPU-based model loading and inference)  

### Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/YaseenMunowwar/Cocolytics-AI.git
   cd Cocolytics-AI
