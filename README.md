# Census Income Prediction – Data-Driven Income Classification (July 2025)

## 🔑 Skills Demonstrated
- Supervised machine learning using scikit-learn
- Logistic Regression and Decision Tree model implementation
- Hyperparameter tuning with `GridSearchCV`
- Data preprocessing with `pandas`, scaling with `StandardScaler`
- Model evaluation using accuracy, precision, recall, and classification reports
- Understanding overfitting vs. generalization through performance tradeoffs

---

## 🧠 Project Summary

This project uses 32,000+ anonymized U.S. Census records to build a machine learning model that predicts whether an individual earns more than $50K annually. The goal was to evaluate and compare different classification algorithms and tune them to achieve strong performance metrics—while understanding the balance between underfitting and overfitting.

---

## 🧪 Approach

1. **Data Preparation**  
   - Loaded data from `censusData.csv`  
   - Converted target column `income_binary` to 0/1 (`<=50K` → 0, `>50K` → 1)  
   - Scaled features using `StandardScaler` for logistic regression

2. **Modeling**  
   - Trained a baseline `LogisticRegression` model (with `max_iter=1000`)  
   - Implemented a Decision Tree classifier and tested various depths and leaf sizes  
   - Used `GridSearchCV` to find optimal tree parameters (`max_depth=10`, `min_samples_leaf=1`, `criterion='gini'`)

3. **Evaluation**  
   - Achieved **~85% accuracy** on both models  
   - Logistic Regression: Recall ≈ **58.5%**, Precision ≈ **71.2%**  
   - Decision Tree (after tuning): Recall ≈ **57.9%**, Precision ≈ **73.3%**

4. **Insights**  
   - Logistic Regression provided strong overall performance with simpler structure  
   - Decision Tree offered more interpretability and competitive precision after tuning  
   - Grid search was key in avoiding overfitting and improving generalization

---

## 📊 Final Results

| Model               | Accuracy | Precision | Recall |
|--------------------|----------|-----------|--------|
| Logistic Regression | 84.7%    | 71.2%     | 58.5%  |
| Decision Tree (tuned) | 85.1% | 73.3%     | 57.9%  |

---

## 📁 Files

- `DefineAndSolveMLProblem.ipynb` – Full notebook with code and evaluation
- `censusData.csv` – Dataset used for training and testing

---

## 🤔 Reflections

This project helped solidify my understanding of:
- Model selection tradeoffs (simplicity vs flexibility)
- Evaluating models beyond accuracy using recall and precision
- Preventing overfitting through hyperparameter tuning
- Data preprocessing techniques essential to model performance
