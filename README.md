# 🏋️ Fitness Classification Using Machine Learning

An intelligent Machine Learning system that predicts whether an individual is **Fit** or **Not Fit** using health and lifestyle attributes. The project compares multiple classification algorithms and identifies the best-performing models for fitness prediction.

---

## 📌 Project Overview

This project uses supervised machine learning techniques to classify individuals based on health-related parameters such as:

- BMI
- Heart Rate
- Blood Pressure
- Sleep Duration
- Daily Steps
- Water Intake
- Calories Burned
- Physical Activity
- Nutrition Quality

The goal is to assist users in understanding their fitness level through data-driven predictions.

---

## 🚀 Features

- Data preprocessing and cleaning
- Missing value handling
- Feature scaling and normalization
- Comparison of multiple ML algorithms
- Model performance evaluation
- Fitness prediction (Fit / Not Fit)
- Visualization of model performance

---

## 🤖 Machine Learning Models

The following models were implemented and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM - RBF Kernel)

### Best Performing Models
- ✅ Random Forest
- ✅ Support Vector Machine (RBF)

---

## 📊 Dataset

Source:
- Kaggle Fitness Dataset

Input Features include:

- Age
- Height
- Weight
- BMI
- Heart Rate
- Blood Pressure
- Sleep Hours
- Nutrition Quality
- Activity Index
- Gender
- Smoking Status

Output:

- Fit
- Not Fit

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure

```
Fitness-Classification/
│
├── dataset/
├── notebooks/
├── models/
├── screenshots/
├── app.py
├── requirements.txt
├── README.md
└── training.py
```

---

## ⚙️ Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Performance Comparison
8. Fitness Prediction

---

## 📈 Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 70% |
| KNN | 67% |
| Random Forest | **72%** |
| SVM (RBF) | **72%** |

Random Forest and SVM demonstrated the best overall performance for fitness classification.

---

## ▶️ Installation

```bash
git clone https://github.com/yourusername/Fitness-Classification.git

cd Fitness-Classification

pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python app.py
```

or

```bash
python training.py
```

---

## 📚 Future Improvements

- Real-time fitness prediction
- Web application using Flask/Streamlit
- Integration with wearable devices
- Deep Learning models
- Personalized fitness recommendations

---

## 👨‍💻 Author

**Mohammed Aakef**

B.Tech Computer Science and Engineering (AI & ML)

SRM Institute of Science and Technology

---

## ⭐ If you found this project useful, please give it a star!
