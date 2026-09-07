import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Remove empty rows in text column
data = data.dropna(subset=['text_'])

# Convert text column to string format (prevents NaN error)
data['text_'] = data['text_'].astype(str)

# Convert label column (CG / OR etc.) into numeric values automatically
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Features & Labels
X = data['text_']
y = data['label']

# Convert text into numeric features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# AdaBoost Model
ada_model = AdaBoostClassifier()
ada_model.fit(X_train, y_train)

# Save models
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(dt_model, open("decision_tree.pkl", "wb"))
pickle.dump(ada_model, open("adaboost.pkl", "wb"))

print("Models trained and saved successfully!")