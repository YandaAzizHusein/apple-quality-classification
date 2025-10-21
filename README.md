# Apple Quality Classification  
**Author:** Yanda Aziz Husein  
**Project Type:** Supervised Machine Learning – Classification  
**Repository:** [github.com/YandaAzizHusein/apple-quality-classification](https://github.com/YandaAzizHusein/apple-quality-classification)

---

## 1. Project Overview  

This project explores the application of supervised machine learning techniques to automate the classification of apple quality (*good* vs *bad*) based on measurable physical and sensory characteristics.  
The objective is to demonstrate how data-driven models can support decision-making in precision agriculture by improving consistency, reducing subjectivity, and increasing operational efficiency.

The project follows a complete end-to-end machine learning workflow covering data preprocessing, feature analysis, model development, evaluation, and performance interpretation.

---

## 2. Business Context  

Manual fruit quality inspection within agricultural supply chains is often inconsistent and time-consuming.  
This project addresses the following business challenges:

- **Inconsistency in human judgment:** Subjective evaluations lead to unreliable quality control results.  
- **Scalability limitations:** Manual inspections cannot handle industrial-scale sorting efficiently.  
- **Lack of automation:** The agricultural sector requires AI-based systems to standardize quality assessments.

By developing a predictive model using measurable attributes such as size, sweetness, juiciness, and ripeness, this study demonstrates how machine learning can enhance post-harvest quality control and strengthen smart-agriculture initiatives.

---

## 3. Dataset Description  

- **Source:** [Apple Quality Dataset – Nidula Elgiriyewithana, Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality)  
- **Total Samples:** 4,000  
- **Attributes:** 9 (8 numerical features + 1 categorical target)  
- **Target Variable:** `Quality` (Good = 1, Bad = 0)  
- **Data Properties:** Clean, normalized, and balanced between classes.

| Feature | Description |
|----------|-------------|
| Size | Physical dimension of the apple |
| Weight | Total mass of the apple |
| Sweetness | Measured sugar level |
| Crunchiness | Texture firmness indicator |
| Juiciness | Moisture content level |
| Ripeness | Degree of fruit maturity |
| Acidity | Acidity level measurement |
| Quality | Target variable indicating good or bad quality |

---

## 4. Methodology  

### Workflow  
The modeling process follows a standard supervised learning pipeline:

**Data → Preprocessing → Modeling → Evaluation → Result**

![Workflow](Asset/Workflow%20Diagram%20Machine%20Learning.png)

### Analytical Stages  

1. **Data Understanding:** Exploratory Data Analysis (EDA), validation of missing values, and target distribution analysis.  
2. **Data Preparation:** Feature cleaning, correlation analysis, label encoding, and data partitioning (train/test = 80:20).  
3. **Modeling:** Implementation of Logistic Regression, Random Forest, and XGBoost classifiers.  
4. **Evaluation:** Comparison using accuracy, precision, recall, F1-score, and confusion matrices.  
5. **Interpretation:** Identification of key predictive features influencing apple quality.

---

## 5. Exploratory Data Analysis  

### Target Distribution  
The dataset shows balanced representation between *good* and *bad* apple quality classes.  

![Distribution](Asset/Distribusi%20Label%20Target%20(Quality).png)

### Feature Correlation  
Correlation analysis indicates four dominant predictors — Size, Sweetness, Juiciness, and Ripeness — strongly correlated with quality.  

![Heatmap](Asset/Heatmap.png)

---

## 6. Modeling and Evaluation  

Three classification models were trained and benchmarked:

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:---------:|:----------:|:--------:|:----------:|
| Logistic Regression | 0.7088 | 0.7098 | 0.7099 | 0.7094 |
| Random Forest | **0.7837** | **0.7871** | **0.7879** | **0.7875** |
| XGBoost | 0.7725 | 0.7770 | 0.7762 | 0.7761 |

**Key Observation:**  
The Random Forest model achieved the best performance overall, indicating superior generalization with minimal tuning effort.

### Confusion Matrices  

| Logistic Regression | Random Forest | XGBoost |
|:-------------------:|:--------------:|:--------:|
| ![Logistic Regression](Asset/Confusion%20Matrix%20-%20Logistic%20Regression.png) | ![Random Forest](Asset/Confusion%20Matrix%20-%20Random%20Forest.png) | ![XGBoost](Asset/Confusion%20Matrix%20-%20XGBoost.png) |

---

## 7. Key Insights  

- The Random Forest classifier obtained **accuracy = 78.37%**, outperforming Logistic Regression and XGBoost.  
- The most influential predictors of apple quality are **Size**, **Sweetness**, **Juiciness**, and **Ripeness**.  
- Machine learning models can replicate expert-level grading decisions objectively and consistently.

---

## 8. Future Work  

Recommended next steps:

1. **Hyperparameter Tuning:** Employ grid search or Bayesian optimization for model refinement.  
2. **Feature Engineering:** Investigate non-linear feature interactions and polynomial transformations.  
3. **Image Integration:** Combine sensor data with RGB or NIR imagery for advanced hybrid modeling.  
4. **Model Deployment:** Deploy as a REST API or web-based dashboard for real-time quality inspection.

---

## 9. Technical Stack  

| Category | Technology |
|-----------|-------------|
| Programming Language | Python 3.10 |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn, xgboost |
| Environment | Google Colab, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## 10. Project Files  

```
├── Apple_Quality_Classification.ipynb
├── Apple_Quality_Classification.py
├── Apple_Quality_Classification_Project_Report.md
└── Asset/
    ├── Workflow Diagram Machine Learning.png
    ├── Heatmap.png
    ├── Distribusi Label Target (Quality).png
    ├── Confusion Matrix - Logistic Regression.png
    ├── Confusion Matrix - Random Forest.png
    └── Confusion Matrix - XGBoost.png
```

Each file supports a different component of the submission:
- **.ipynb** — main notebook containing executed code and narrative.  
- **.py** — clean script version for reproducibility.  
- **.md** — project report formatted for review and portfolio presentation.  

---

## 11. Author  

**Yanda Aziz Husein**  
Machine Learning and Data Science Enthusiast  
[GitHub Profile](https://github.com/YandaAzizHusein)  

---

> *This project demonstrates a practical implementation of machine learning for agricultural product quality control, emphasizing reproducibility, interpretability, and real-world applicability.*
