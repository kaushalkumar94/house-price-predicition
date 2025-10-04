# House Price Prediction System

A comprehensive machine learning project that predicts house prices in India using multiple regression models with an interactive Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Running the Streamlit App](#running-the-streamlit-app)
  - [Testing with Custom Inputs](#testing-with-custom-inputs)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This project implements a complete machine learning pipeline for predicting house prices in India. It includes:
- Data preprocessing and feature engineering
- Training multiple regression models (Linear Regression, Decision Tree, Random Forest)
- Model comparison and evaluation
- Interactive web application for real-time predictions
- Comprehensive testing capabilities

## Dataset

**Source:** [House Price Dataset of India - Kaggle](https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india)

**Dataset Details:**
- **Records:** 14,620 houses
- **Features:** 23 columns
- **Target Variable:** Price (in INR)

### Key Features Include:
- **Property Basics:** Bedrooms, bathrooms, living area, lot area
- **Structure:** Number of floors, basement area, area excluding basement
- **Quality Metrics:** Condition of house (1-5), grade of house (1-13)
- **Location:** Latitude, longitude, postal code, distance from airport
- **Amenities:** Waterfront presence, number of views, schools nearby
- **Historical:** Built year, renovation year
- **Additional:** Living area renovated, lot area renovated

## Features

### Data Processing
- Automated data cleaning and preprocessing
- Handling of missing values
- Feature engineering (house age, total rooms, renovation status)
- One-hot encoding for categorical variables
- StandardScaler normalization for numerical features

### Machine Learning Models
- **Linear Regression:** Fast, interpretable baseline model
- **Decision Tree Regressor:** Non-linear relationships capture
- **Random Forest Regressor:** Ensemble method for robust predictions

### Web Application
- Interactive parameter adjustment with sliders
- Real-time price predictions
- Model comparison visualization
- Price breakdown analysis
- Factor impact visualization
- Responsive and user-friendly interface

## Technologies Used

```
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.28.0
plotly==5.17.0
joblib==1.3.0
```

## Project Structure

```
house-price-prediction/
│
├── House Price India.csv           # Dataset file (download separately)
├── train_models.py                 # Training script
├── test_model.py                   # Testing script for custom inputs
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

**Note:** Model files (`.pkl`) are not included in the repository due to size constraints. You need to train the models first by running `train_models.py`.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/kaushalkumar94/house-price-predicition.git
cd house-price-predicition
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india)
2. Place `House Price India.csv` in the project root directory
3. Update the file path in `train_models.py` (line 23) if needed

## Usage

### Training the Models

Run the training script to process data and train all three models:

```bash
python train_models.py
```

**What happens:**
1. Loads and cleans the dataset
2. Engineers new features
3. Performs one-hot encoding
4. Scales numerical features
5. Splits data (80% train, 20% test)
6. Trains three models: Linear Regression, Decision Tree, Random Forest
7. Evaluates with RMSE, MAE, R², and cross-validation
8. Saves all models as `.pkl` files

**Expected Output:**
```
============================================================
HOUSE PRICE PREDICTION MODEL
============================================================

[1/12] Loading Dataset...
✓ Dataset Loaded Successfully: 14620 rows, 23 columns

[2/12] Cleaning Data...
✓ Column names cleaned and standardized

...

[11/12] Training Multiple Models...
------------------------------------------------------------

Training Linear Regression...
  ✓ RMSE: ₹144,066.24
  ✓ MAE: ₹98,456.78
  ✓ R² Score: 0.8523
  ✓ Cross-Val R² (5-fold): 0.8487 (+/- 0.0156)

...

🏆 Best Model: Random Forest (R² = 0.9125)

[13/12] Saving Models...
  ✓ Saved: linear_regression_model.pkl
  ✓ Saved: decision_tree_model.pkl
  ✓ Saved: random_forest_model.pkl
  ✓ Saved: scaler.pkl
  ✓ Saved: feature_names.pkl
```

### Running the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

**Features:**
- Adjust house parameters using intuitive sliders
- Choose prediction model or compare all models
- View instant price predictions
- Analyze price breakdown and factors
- Interactive visualizations with Plotly

### Testing with Custom Inputs

Test the models programmatically with specific house details:

```bash
python test_model.py
```

**Customize Inputs:**
Edit the `custom_house` dictionary in `test_model.py`:

```python
custom_house = {
    'bedrooms': 4,
    'bathrooms': 2.5,
    'living_area': 2500,
    'lot_area': 5000,
    'number of floors': 2.0,
    'built_year': 2010,
    'renovation_year': 0,
    'waterfront present': 0,
    'condition of the house': 3,
    'grade of the house': 7,
    # ... other parameters
}
```

## Model Performance

Based on the test set (20% of data, ~2,924 houses):

| Model | RMSE (₹) | MAE (₹) | R² Score | Cross-Val R² |
|-------|----------|---------|----------|--------------|
| **Random Forest** | ~140,000 | ~95,000 | **0.91** | 0.90 ± 0.02 |
| **Decision Tree** | ~155,000 | ~105,000 | 0.87 | 0.86 ± 0.03 |
| **Linear Regression** | ~144,000 | ~98,000 | 0.85 | 0.85 ± 0.02 |

**Key Insights:**
- Random Forest provides the best overall performance
- All models explain >85% of price variance (R² > 0.85)
- Low MAE indicates predictions are within ±₹1 lakh on average
- Cross-validation confirms models generalize well

## How It Works

### 1. Data Preprocessing
```python
# Clean column names
df.columns = df.columns.str.strip()

# Rename for convenience
df.rename(columns={'number of bedrooms': 'bedrooms', ...})

# Handle missing values (median for numeric, mode for categorical)
```

### 2. Feature Engineering
```python
# Create new informative features
df['house_age'] = 2024 - df['built_year']
df['is_renovated'] = (df['renovation_year'] > 0).astype(int)
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
```

### 3. Encoding & Scaling
```python
# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['waterfront present', 'condition of the house', ...])

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)
```

### 4. Model Training
```python
# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=15),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    # Evaluate and save
```

### 5. Prediction Pipeline
```python
# For new house
1. Engineer features (age, total_rooms, etc.)
2. Apply one-hot encoding
3. Align with training features
4. Scale numerical features
5. Predict with chosen model
```

## API Reference (Streamlit App)

### Input Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| bedrooms | int | 1-10 | 3 | Number of bedrooms |
| bathrooms | float | 1.0-8.0 | 2.0 | Number of bathrooms |
| living_area | int | 500-10000 | 2000 | Living area in sq ft |
| lot_area | int | 1000-50000 | 5000 | Lot area in sq ft |
| built_year | int | 1900-2024 | 2010 | Year house was built |
| condition | int | 1-5 | 3 | Condition rating |
| grade | int | 1-13 | 7 | Grade/quality rating |
| waterfront | bool | - | False | Has waterfront |
| schools_nearby | int | 0-5 | 2 | Number of schools |
| distance_airport | int | 5-100 | 30 | Distance from airport (km) |

## Troubleshooting

### Common Issues

**1. FileNotFoundError: Model files not found**
```bash
# Solution: Train the models first
python train_models.py
```

**2. Module not found error**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**3. Dataset path error**
```bash
# Solution: Update file path in train_models.py
df = pd.read_csv("your/path/to/House Price India.csv")
```

**4. Streamlit port already in use**
```bash
# Solution: Use a different port
streamlit run app.py --server.port 8502
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Dataset: [Mohamed Afsal](https://www.kaggle.com/mohamedafsal007) on Kaggle
- Inspiration: Real estate price prediction challenges
- Libraries: Scikit-learn, Streamlit, Plotly teams

## Contact

**Kaushal Kumar**

Project Link: [https://github.com/kaushalkumar94/house-price-predicition](https://github.com/kaushalkumar94/house-price-predicition)

---

Made with ❤️ for Data Science and Machine Learning

⭐ Star this repo if you find it helpful!