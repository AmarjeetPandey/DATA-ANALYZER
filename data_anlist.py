import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns
st.title("📊 PDF TABLE ANLYZER")

pdf_file = st.file_uploader("Upload PDF File", type=["pdf"])

if pdf_file is not None:
    with pdfplumber.open(pdf_file) as pdf:
        all_tables = []

        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables.append(df)

    if all_tables:
        df = pd.concat(all_tables, ignore_index=True)

        st.success("Table Extracted Successfully ✅")
        st.write(df.head())
        
        if st.button("Show Shape"):
         st.write("Rows & Columns:", df.shape)

        if st.button("Show Columns"):
         st.write(df.columns)

        if st.button("Describe Data"):
         st.write(df.describe())

        if st.button("Check Null Values"):
         st.write(df.isnull().sum())
    
        if st.button("show complete overview of data"):
         st.write(df.info()) 
        
        if st.button("show indexing in the dataset "):
         st.write(df.index)
        
        if st.button("show datatypes of dataset"):
         st.write(df.dtypes)
        
        if st.button("show the upper 5 lines of dataset"):
         st.write(df.head())
        
        if st.button("show the lower 5 lines of dataset"):
         st.write(df.tail())
        
        if st.button("show the random rows of dataset"):
         st.write(df.sample())                   
        
        if st.button("show the overview of numeric columns of dataset"):
         st.write(df.describe())
        
        if st.button("to check the the only complete filed rows "):
         st.write(df.notnull())
        
        if st.button("to chech unique value in each column"):
         st.write(df.nunique())
        
        if st.button("repeated value in dataset"):
         st.write(df.value_counts()) 
        
        if st.button("to find duplicate value in dataset"):
         st.write(df.duplicated())
        
        if st.button("remove duplicate value from your dataset"):
         st.write(df.drop_duplicates())  
    


st.title("📊 CSV FILE DATA ANLYZER")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin-1")
    
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Buttons
    if st.button("Row & column"):
        st.write("Rows & Columns:", df.shape)

    if st.button("Columns name"):
        st.write(df.columns)

    if st.button("Describe data"):
        st.write(df.describe())

    if st.button("missing Values"):
        st.write(df.isnull().sum())
    
    if st.button("show overview of data"):
        st.write(df.info()) 
        
    if st.button("show index in the dataset "):
        st.write(df.index)
        
    if st.button("show datatypes"):
        st.write(df.dtypes)
        
    if st.button("show the upper 5 lines"):
        st.write(df.head())
        
    if st.button("show the lower 5 lines"):
        st.write(df.tail())
        
    if st.button("random rows of dataset"):
        st.write(df.sample())                   
        
    if st.button("show numeric overview"):
        st.write(df.describe())
        
    if st.button("shows only complete filed rows "):
        st.write(df.notnull())
        
    if st.button("unique value in of column"):
        st.write(df.nunique())
        
    if st.button("repeated value in column"):
        st.write(df.value_counts()) 
        
    if st.button("shows duplicate value in column"):
        st.write(df.duplicated())
        
    if st.button("remove duplicate value in column"):
        st.write(df.drop_duplicates())

if st.button("Show Gender Count"):
    fig, ax = plt.subplots()

    ax = sns.countplot(x='Gender', data=df, ax=ax)

    for bars in ax.containers:
        ax.bar_label(bars)

    st.pyplot(fig)

if st.button("show age group count"): 
    fig, ax = plt.subplots() 
     
    ax = sns.countplot(data = df, x = 'Age Group', hue = 'Gender')

    for bars in ax.containers:
       ax.bar_label(bars)
       
    st.pyplot(fig)
    
if st.button("Show Sales by Age Group"):
    sales_age = df.groupby(['Age Group'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x = 'Age Group',y= 'Amount' ,data = sales_age, ax=ax)
    st.pyplot(fig)
    
if st.button("show top 10 state with highest sales"):
   sales_state = df.groupby(['State'], as_index=False)['Orders'].sum().sort_values(by='Orders', ascending=False).head(10) 
   
   fig, ax = plt.subplots()
   sns.set(rc={'figure.figsize':(15,5)})
   sns.barplot(data = sales_state, x = 'State',y= 'Orders', ax=ax)
   st.pyplot(fig)
    
if st.button("show top 10 state with highest sales amount "):
   sales_state = df.groupby(['State'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False).head(10)

   fig, ax = plt.subplots()
   sns.set(rc={'figure.figsize':(15,5)})
   sns.barplot(data = sales_state, x = 'State',y= 'Amount')
   st.pyplot(fig)    
   
if st.button("show marital status count"):

    sales_state = df.groupby(['Marital_Status','Gender'], as_index=False)['Amount'].sum()

    fig, ax = plt.subplots()

    sns.barplot(data=sales_state, x='Marital_Status', y='Amount', hue='Gender', ax=ax)

    st.pyplot(fig)
    
# ===================== FINAL STABLE ML RESEARCH PAPER ANALYZER =====================

import nltk
import pdfplumber
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

nltk.download('punkt')
nltk.download('punkt_tab')

st.title("🧠 Advanced Research Paper Analyzer (Pure ML - Stable)")

uploaded_doc = st.file_uploader(
    "Upload Research Paper (PDF / TXT)", 
    type=["pdf", "txt"]
)

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa_model = SentenceTransformer('all-MiniLM-L6-v2')
    return summarizer, qa_model

summarizer, qa_model = load_models()

# Extract text
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    else:
        return file.read().decode("utf-8", errors="ignore")

# Chunking
def chunk_text(text, max_chunk=500):
    words = text.split()
    for i in range(0, len(words), max_chunk):
        yield " ".join(words[i:i+max_chunk])

# Summary
def generate_summary(text):
    chunks = list(chunk_text(text))
    final_summary = ""

    for chunk in chunks[:5]:
        summary = summarizer(chunk, max_length=120, min_length=40, do_sample=False)
        final_summary += summary[0]['summary_text'] + " "

    return final_summary

# Keywords (TF-IDF based)
def extract_keywords(text):
    vec = TfidfVectorizer(stop_words='english', max_features=10)
    X = vec.fit_transform([text])
    return vec.get_feature_names_out()

# Sections
def extract_sections(text):
    sections = {
        "Abstract": "",
        "Introduction": "",
        "Methodology": "",
        "Conclusion": ""
    }

    lines = text.split("\n")
    current = None

    for line in lines:
        l = line.lower()

        if "abstract" in l:
            current = "Abstract"
        elif "introduction" in l:
            current = "Introduction"
        elif "method" in l:
            current = "Methodology"
        elif "conclusion" in l:
            current = "Conclusion"

        if current:
            sections[current] += " " + line

    return sections

# Main
if uploaded_doc is not None:
    text = extract_text(uploaded_doc)

    if len(text.strip()) == 0:
        st.error("❌ No readable text found in document")
    else:
        st.success("Document Loaded ✅")

        st.subheader("📄 Preview")
        st.write(text[:1000])

        if st.button("🚀 Full AI Analysis"):

            with st.spinner("Running ML Models..."):

                summary = generate_summary(text)
                keywords = extract_keywords(text)
                sections = extract_sections(text)

            st.subheader("📌 Summary")
            st.write(summary)

            st.subheader("🔑 Keywords")
            st.write(keywords)

            st.subheader("📂 Sections")

            for sec in sections:
                st.write(f"**{sec}:**")
                st.write(sections[sec][:500])

        # 🔥 ADVANCED Q&A (semantic search)
        user_q = st.text_input("💬 Ask anything from paper")

        if user_q:
            sentences = nltk.sent_tokenize(text)

            if len(sentences) == 0:
                st.error("❌ Text processing failed")
            else:
                try:
                    sentence_embeddings = qa_model.encode(sentences, convert_to_tensor=True)
                    query_embedding = qa_model.encode(user_q, convert_to_tensor=True)

                    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
                    top_results = torch.topk(scores, k=3)

                    st.subheader("📖 Answer")

                    for idx in top_results[1]:
                        st.write(sentences[idx])

                except Exception as e:
                    st.error(f"Error: {e}")