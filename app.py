import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (only once)
nltk.download("stopwords")
nltk.download("punkt")

# Load model and vectorizer
try:
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error("❌ Failed to load model/vectorizer. Please check your files.")
    st.stop()

# Cleaning function
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **real** or **fake**.")

user_input = st.text_area("✍️ Paste your news article or headline:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip():
        cleaned_text = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]
        if prediction == 1:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("🚨 This appears to be **Fake News**.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
