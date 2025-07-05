# 🧠 Intelligent Complaint Analysis with RAG

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using real-world consumer complaints from the CFPB dataset. The system is designed to semantically search and retrieve complaint narratives using modern NLP techniques like embedding and vector databases (FAISS/ChromaDB).

---

## 🚀 Project Objectives

- Clean and preprocess large-scale complaint data
- Analyze trends and patterns in complaints
- Break long narratives into meaningful chunks
- Convert text to semantic vector representations
- Persist the vector index for use in downstream GenAI apps (RAG)

---

## 📂 Project Structure

intelligent-complaint-analysis/
├── data/
│ ├── raw/ # Original CFPB dataset
│ ├── processed/ # Cleaned data
│ └── filtered_complaints.csv # Final filtered dataset used for RAG
├── notebooks/
│ └── task_1_eda_preprocessing.ipynb # EDA + preprocessing notebook
├── src/
│ ├── eda_preprocessing.py # Python script for Task 1
│ ├── chunk_and_embed.py # Python script for Task 2
│ └── config.py # Chunking and embedding settings
├── vector_store/
│ └── faiss_index/ # FAISS index (not tracked by Git)
├── .gitignore
├── requirements.txt
└── README.md



## ✅ Tasks Completed

### 📊 Task 1: EDA and Data Preprocessing

- Loaded the full CFPB complaint dataset
- Conducted analysis on:
  - Distribution of complaints by product
  - Length distribution of complaint narratives
  - Number of narratives with and without text
- Filtered for the following products:
  - `Credit card`, `Personal loan`, `Buy Now, Pay Later`, `Savings account`, `Money transfers`
- Cleaned complaint narratives:
  - Lowercased
  - Removed boilerplate phrases and special characters
  - Normalized whitespace

🔖 **Output**: `data/filtered_complaints.csv`



### 🧩 Task 2: Chunking, Embedding & Vector Indexing

- Implemented a chunking strategy using `RecursiveCharacterTextSplitter`
  - Tuned chunk size and overlap to balance retrieval quality
- Used `sentence-transformers/all-MiniLM-L6-v2` for sentence-level embeddings
- Stored chunks and metadata in a FAISS vector store
- Metadata includes:
  - Complaint ID
  - Product category
  - Original text chunk

🧠 **Vector Store**: Persisted in `vector_store/faiss_index/` (ignored in Git)



## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core implementation |
| **LangChain** | Text splitting and vector store API |
| **FAISS** | High-speed vector similarity search |
| **sentence-transformers** | Pretrained embedding models |
| **pandas, matplotlib** | Data handling and visualization |
| **Jupyter Notebook** | Exploratory work and reporting |



## 🛡️ Git and Large Files

The FAISS index exceeds GitHub's 100MB limit and is ignored via `.gitignore`. If needed for remote access, use [Git LFS](https://git-lfs.github.com/) or regenerate the index with the provided script.


# Ignore large vector store
echo "vector_store/" >> .gitignore
📦 Setup Instructions

# Clone the repo
git clone https://github.com/tsegabogale24/intelligent-complaint-analysis
cd intelligent-complaint-analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
