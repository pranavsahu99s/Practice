# Cosmic Classifier 🌌

![Cosmic Banner](https://via.placeholder.com/800x200?text=Cosmic+Classifier)

## Introduction

Welcome to the Cosmic Classifier project, an advanced machine learning solution developed for the Galactic Classification Challenge (GCC) 2025 hosted by IIT Roorkee. This project tackles the fascinating challenge of classifying exoplanets based on their potential for human survival and resource availability.

In the year 2547, Dr. Klaus Reinhardt transmitted crucial planetary classification data before being consumed by a black hole. Unfortunately, the transmission was partially corrupted due to gravitational interference. Our mission is to decode this damaged dataset and accurately classify planets to secure humanity's future among the stars.

This repository contains the complete solution including data preprocessing, exploratory data analysis, model training, evaluation, and prediction pipeline.

## 🚀 Project Overview

The Cosmic Classifier uses multiple machine learning models to categorize planets into 10 distinct classes based on 10 planetary attributes. We've implemented a robust comparison framework to evaluate several classification algorithms and select the most effective approach.

### 🌠 The Challenge

- **Classification Task**: Predict planet types using 10 attributes including atmospheric density, surface temperature, gravity, water content, etc.
- **Noisy Data**: Handle transmission interference and missing values in the dataset
- **High Stakes**: Accuracy is crucial as humanity's survival depends on correct classification

## 📊 Dataset Description

The dataset contains information about planets with the following features:

1. **Atmospheric Density**: Measure of the planet's atmosphere thickness (kg/m³)
2. **Surface Temperature**: Average surface temperature (Kelvin)
3. **Gravity**: Surface gravity (m/s²)
4. **Water Content**: Percentage of surface covered by water (0-100%)
5. **Mineral Abundance**: Availability of valuable minerals (scale 0-1)
6. **Orbital Period**: Time to orbit its star (Earth days)
7. **Proximity to Star**: Distance from the planet to its star (AU)
8. **Magnetic Field Strength**: Measure of the planet's magnetic field (Tesla)
9. **Radiation Levels**: Average radiation on the planet's surface (Sieverts/year)
10. **Atmospheric Composition Index**: Suitability of atmosphere for human life (scale 0-1)

### Target Classes

Planets are classified into 10 types (in German):
1. Bewohnbar
2. Terraformierbar
3. Rohstoffreich
4. Wissenschaftlich
5. Gasriese
6. Wüstenplanet
7. Eiswelt
8. Toxischetmosäre
9. Hohestrahlung
10. Toterahswelt

## 🧠 Models Implemented

Our solution compares multiple machine learning approaches to find the optimal model:

### 1. Logistic Regression
A linear model that serves as our baseline classifier. Despite its simplicity, it provides a solid foundation for more complex models and helps identify linear relationships in the data.

### 2. Decision Tree Classifier
A highly interpretable model that creates decision rules based on feature values. This model is particularly valuable for the challenge as it provides clear rationale for classifications, which is crucial when the stakes are humanity's survival.

### 3. Random Forest Classifier
An ensemble of decision trees that improves prediction accuracy by combining multiple models. This approach helps mitigate overfitting while maintaining the ability to capture complex patterns in planetary data.

### 4. Gradient Boosting Classifier
A sequential ensemble method that builds trees one after another, with each tree correcting the errors of previous trees. This powerful approach excels at handling the complex relationships between planetary features.

### 5. AdaBoost Classifier
An adaptive boosting algorithm that adjusts the importance of training examples after each iteration, focusing on difficult-to-classify planets. This approach helps identify edge cases in the dataset.

### 6. XGBoost Classifier
An optimized implementation of gradient boosting that combines regularization with efficient computation. This state-of-the-art algorithm often achieves superior performance on structured data problems.

## 🛠️ Technical Features

### Preprocessing Pipeline
- Handling missing values and outliers in transmission data
- Feature scaling and normalization
- Feature engineering to extract meaningful patterns

### Model Training Framework
- Cross-validation strategy to ensure robust evaluation
- Hyperparameter tuning for optimal model performance
- Ensemble methods to leverage the strengths of multiple models

### Evaluation Metrics
- Accuracy (primary metric for the competition)
- F1 score (weighted average)
- Precision and recall
- ROC AUC scores with special handling for multiclass classification

### Deployment
- Model serialization using pickle for easy deployment
- Prediction pipeline for seamless inference on new planetary data

## 📈 Results

Our comprehensive model evaluation reveals detailed performance metrics for each classifier on both training and test datasets. The Decision Tree model was selected for final deployment based on its balance of performance, interpretability, and computational efficiency.

> Note: Full performance metrics and comparison charts are available in the Jupyter notebook documentation.

## 🔧 Installation and Usage

### Prerequisites
- Python 3.8+
- Required packages: numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/cosmic-classifier.git
cd cosmic-classifier

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Running the Solution
```bash
# Run the Jupyter notebook
jupyter notebook Cosmic_Classifier_Solution.ipynb
```

### Making Predictions on New Data
```python
import pickle

# Load the trained model
with open("decision_tree_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Make predictions on new data
predictions = model.predict(new_data)
```

## 📁 Repository Structure

```
cosmic-classifier/
├── data/
│   ├── train.csv          # Training dataset
│   └── test.csv           # Test dataset (without labels)
├── notebooks/
│   ├── EDA.ipynb          # Exploratory Data Analysis
│   └── Model_Training.ipynb # Model development and evaluation
├── models/
│   └── decision_tree_model.pkl # Serialized model
├── results/
│   └── model_performance.csv  # Performance metrics for all models
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preparation scripts
│   ├── feature_engineering.py # Feature creation and transformation
│   ├── model_training.py      # Model training utilities
│   └── evaluation.py          # Metrics calculation and visualization
├── README.md              # Project documentation
├── requirements.txt       # Project dependencies
└── Cosmic_Classifier_Solution.ipynb # Main solution notebook
```

## 🎯 Competition Information

This project was developed for the Galactic Classification Challenge (GCC) organized by Cognizance, IIT Roorkee's annual technical festival. The competition consisted of two rounds:

1. **Round 1 (Code Submission)**: Qualification round requiring a Python notebook solution
2. **Round 2 (Offline Testing)**: On-site evaluation at IIT Roorkee with new test data

## 👥 Team

- [Team Member 1](https://github.com/member1) - Role/Contribution
- [Team Member 2](https://github.com/member2) - Role/Contribution
- [Team Member 3](https://github.com/member3) - Role/Contribution

## 📚 References

1. Research papers and articles that influenced our approach
2. Key machine learning resources that supported the project
3. Documentation for libraries and frameworks used

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Disclaimer: This project was created for educational purposes as part of the Galactic Classification Challenge (GCC) 2025. The fictional narrative about Dr. Klaus Reinhardt and humanity's future is part of the competition's theme.*
