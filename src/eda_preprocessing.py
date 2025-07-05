import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import os
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define target products
TARGET_PRODUCTS = [
    'Credit card',
    'Personal loan',
    'Buy Now, Pay Later (BNPL)',
    'Savings account',
    'Money transfers'
]

# ----------------------------
# Load and Inspect Data
# ----------------------------

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def initial_eda(df):
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nColumns:", df.columns.tolist())
    print("\nSample rows:\n", df.head())

# ----------------------------
# Visualization
# ----------------------------

def plot_complaint_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Product', data=df, order=df['Product'].value_counts().index)
    plt.title('Distribution of Complaints by Product')
    plt.xlabel('Count')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.show()

def analyze_narrative_length(df):
    df['Narrative Length'] = df['Consumer complaint narrative'].dropna().apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Narrative Length'], bins=50, kde=True)
    plt.title('Distribution of Narrative Length')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    print("\nNarrative length stats:")
    print(df['Narrative Length'].describe())

def count_narratives(df):
    with_narrative = df['Consumer complaint narrative'].notna().sum()
    without_narrative = df['Consumer complaint narrative'].isna().sum()
    print(f"\nComplaints with narrative: {with_narrative}")
    print(f"Complaints without narrative: {without_narrative}")

# ----------------------------
# Filtering & Cleaning
# ----------------------------

def filter_data(df):
    df_filtered = df[df['Product'].isin(TARGET_PRODUCTS)]
    df_filtered = df_filtered[df_filtered['Consumer complaint narrative'].notna()].copy()
    print("\nFiltered shape:", df_filtered.shape)
    return df_filtered

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.strip()

def clean_narratives(df, text_col='Consumer complaint narrative'):
    df['Cleaned Narrative'] = df[text_col].apply(clean_text)
    return df

# ----------------------------
# Boilerplate Detection & Removal
# ----------------------------

def detect_boilerplate_phrases(docs, top_n=10, min_df=0.2, max_df=0.9):
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 5),
        stop_words='english',
        min_df=min_df,
        max_df=max_df
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    phrases = np.array(vectorizer.get_feature_names_out())
    top_phrases = phrases[np.argsort(-tfidf_scores)][:top_n]
    return top_phrases.tolist()

def remove_boilerplate_phrases(text, boilerplate_phrases):
    for phrase in boilerplate_phrases:
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        text = pattern.sub('', text)
    return text.strip()

# ----------------------------
# Save Cleaned Dataset
# ----------------------------

def save_cleaned_data(df, path='../data/processed/filtered_complaints.csv'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path}")
