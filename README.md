# fake-review-detection-usingML
AI-powered fake review detection using ML, TF-IDF, Decision Tree, AdaBoost, and Flask.
# 🕵️ Fake Review Detection

An AI-powered web application that detects whether a product review is **genuine or fake** using Natural Language Processing (NLP) and Machine Learning.

The project uses **TF-IDF feature extraction** and an **AdaBoost classification model** to analyze review text. It also includes a rule-based system to identify meaningless or suspicious input before making a prediction.

## ✨ Features

- 🔍 Detects fake and genuine reviews
- 📝 Accepts review text through a web interface
- 🧠 Uses TF-IDF to convert text into numerical features
- 🤖 Trains Decision Tree and AdaBoost models
- 🛡️ Detects meaningless or suspicious text
- 🌐 Flask-based web application
- 💾 Saves trained models using Pickle

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Pandas**
- **Scikit-learn**
- **NLTK**
- **TF-IDF Vectorizer**
- **Decision Tree Classifier**
- **AdaBoost Classifier**
- **Pickle**

## 📂 Project Structure

```text
Fake-Review-Detection/
│
├── app.py                 # Flask application
├── model.py               # Model training script
├── requirements.txt       # Required Python libraries
├── dataset.csv            # Dataset used for training
│
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── decision_tree.pkl      # Saved Decision Tree model
├── adaboost.pkl           # Saved AdaBoost model
│
├── templates/
│   └── index.html         # Web interface
│
└── static/
    └── style.css          # Styling files
⚙️ How It Works
1. Data Preparation

The dataset is loaded using Pandas. Empty review texts are removed, and the review text is converted into string format.

2. Text Feature Extraction

The review text is converted into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency).

3. Model Training

The dataset is divided into training and testing sets. Two machine learning models are trained:

Decision Tree Classifier
AdaBoost Classifier
4. Model Saving

The trained models and TF-IDF vectorizer are saved as .pkl files using Pickle.

5. Review Prediction

The Flask application accepts a review from the user, checks whether the text is meaningful, and then uses the saved AdaBoost model to predict whether the review is fake or genuine.
.

🎯 Objective

The objective of this project is to use machine learning and NLP techniques to identify misleading product reviews and improve trust in online review systems.

🔮 Future Enhancements
Improve prediction accuracy with more training data
Compare additional machine learning algorithms
Add model evaluation metrics
Deploy the application online
Support multiple languages
Improve the user interface
👩‍💻 Author

Lavanya Siripurapu

B.Tech Computer Science Student
VIT-AP University
