# 🤖 Sentimental Analysis App

A **Streamlit-based web application** that uses a **BERT-based multilingual model** to perform **Sentiment Analysis** on text input or bulk data (CSV).  
Supports **English, French, Spanish, German, Italian, and Dutch**.  

---

## 📌 **Features**
✔ **Single Text Analysis** – Enter text and get:
- Sentiment rating (⭐ 1 to 5 stars)
- Confidence score with a progress bar
- WordCloud visualization of input text

✔ **Bulk Analysis (Upload CSV)** – Upload a CSV with a `text` column and get:
- Sentiment rating for each text
- Confidence percentage
- Sentiment distribution chart

✔ **Modern UI**
- Tabs for single & bulk analysis
- Custom styling for stars
- Emoji favicon (🤖)
- Sidebar instructions

---

## 🛠 **Tech Stack**
- **Python**
- **Streamlit** (Frontend)
- **Hugging Face Transformers** (BERT model)
- **Matplotlib & WordCloud** (Visualization)
- **Pandas** (Data handling)

---

## 🚀 **How to Run Locally**
1. **Clone this repo**
```bash
git clone https://github.com/<your-username>/sentimental-analysis-app.git
cd sentimental-analysis-app
