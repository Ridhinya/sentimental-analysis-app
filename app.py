import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

st.set_page_config(
    page_title="Sentimental Analysis",
    page_icon="🤖",  
    layout="centered"
)


# Custom CSS for better styling
st.markdown("""
    <style>
        .stars {
            font-size: 30px;
            color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

# Main title
st.title("🤖 Sentimental Analysis")

# Sidebar content
st.sidebar.title("📊 Sentimental Analysis")
st.sidebar.write("""
Analyze text sentiment using a **BERT-based multilingual model**.
- **1 star = Very Negative**
- **5 stars = Very Positive**
Supports English, French, Spanish, German, Italian, Dutch.
""")
st.sidebar.info("Upload a CSV file with a column named `text` for bulk analysis.")

# Load BERT model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = load_model()

# Function to get sentiment with stars and confidence
def get_sentiment(text):
    result = classifier(text)[0]
    label = result['label']  # e.g., "4 stars"
    score = round(result['score'], 3)

    stars_num = int(label.split()[0])
    star_display = "⭐" * stars_num

    return star_display, stars_num, score

# Tabs for UI
tab1, tab2 = st.tabs(["🔍 Single Text Analysis", "📂 Bulk Analysis (Upload CSV)"])

# --- Single Text Tab ---
with tab1:
    st.header("Single Text Analysis")
    user_text = st.text_area("Type your text here:")

    if st.button("Analyze"):
        if user_text.strip():
            stars_display, stars_num, score = get_sentiment(user_text)
            confidence_percentage = int(score * 100)

            # Display results vertically
            st.subheader("Rating:")
            st.markdown(f"<div class='stars'>{stars_display}</div>", unsafe_allow_html=True)

            st.subheader("Confidence:")
            st.write(f"{confidence_percentage}%")
            st.progress(confidence_percentage)

            st.subheader("WordCloud:")
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.error("Please enter some text before analyzing.")

# --- Bulk Analysis Tab ---
with tab2:
    st.header("Bulk Analysis (Upload CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file with a column named 'text'", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            st.error("CSV must have a column named 'text'")
        else:
            st.write("Uploaded Data:", df.head())

            df['StarsDisplay'], df['StarsNum'], df['ConfidenceRaw'] = zip(*df['text'].apply(get_sentiment))
            df['Confidence'] = (df['ConfidenceRaw'] * 100).astype(int).astype(str) + "%"

            st.subheader("Analysis Result:")
            st.write(df[['text', 'StarsDisplay', 'Confidence']])

            st.subheader("Sentiment Distribution:")
            sentiment_counts = df['StarsNum'].value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=['red', 'orange', 'gray', 'blue', 'green'])
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel("Rating (Stars)")
            ax.set_ylabel("Count")
            ax.set_title("Sentiment Distribution (1 = Very Negative, 5 = Very Positive)")
            st.pyplot(fig)
