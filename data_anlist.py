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