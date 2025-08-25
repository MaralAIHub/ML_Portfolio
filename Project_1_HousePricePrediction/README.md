# 🏡 House Prices Prediction: Model Comparison

This project compares several machine learning models for predicting house prices, using multiple feature selection strategies and preprocessing techniques.  
The goal is to evaluate models not only in terms of accuracy (R²) but also error metrics (MAE, RMSE).

---

## 📊 Models Evaluated
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Machine (SVM)
- Multi-layer Perceptron (MLP)

---

## 🎯 Feature Strategies
1. **All Features** – using all available features  
2. **Top10 Correlation** – top 10 features most correlated with the target  
3. **Top10 Importance** – top 10 features selected by RandomForest feature importance  
4. **PCA (10)** – dimensionality reduction to 10 components  

---

## 📈 Evaluation Metrics
- **R² Score** – proportion of variance explained by the model  
- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  

---

## 🔥 Results Summary
- Gradient Boosting showed the best performance with all features.  
- PCA reduced dimensionality but slightly decreased performance.  
- SVM performed poorly on this dataset.  
- Random Forest and Gradient Boosting are competitive, robust choices.  

---

## 📊 Visualizations
The notebook generates:
- **Bar plots** comparing R² scores across models and strategies  
- **Heatmaps** for R², MAE, and RMSE  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MaralAIHub/ML_Portfolio.git
   cd <ML_Portfolio>
