import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load pickled vectorizers & models
with open("vectorizer_nb.pkl", "rb") as f:
    nb_vectorizer = pickle.load(f)

with open("spam_classifier_nb.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("vectorizer_lg.pkl", "rb") as f:
    lr_vectorizer = pickle.load(f)

with open("spam_classifier_lg.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Function to predict message category
def predict_message(text, model, vectorizer):
    processed_text = preprocess_text(text)
    transformed_text = vectorizer.transform([processed_text])
    prediction = model.predict(transformed_text)[0]
    return prediction

# Streamlit UI
st.title("📩 Spam Classifier App")
st.write("Enter a message below to check if it's **Spam** or **Ham**.")

# Text input
user_input = st.text_area("Enter your message here:", "")

# Model selection
model_choice = st.radio("Choose a model:", ["Naïve Bayes", "Logistic Regression"])

# Predict button
if st.button("Classify Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        if model_choice == "Naïve Bayes":
            result = predict_message(user_input, nb_model, nb_vectorizer)
        else:
            result = predict_message(user_input, lr_model, lr_vectorizer)

        # Display result
        if result == "spam":
            st.error("🚨 This message is **Spam!**")
        else:
            st.success("✅ This message is **Ham (Not Spam).**")

