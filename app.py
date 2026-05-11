import streamlit as st
import pandas as pd

# Title
st.title("🚗 Advanced Time Series Clustering with DTW")

# Load dataset
df = pd.read_csv("dataset.csv")

# Show dataset
st.subheader("Dataset Preview")

st.dataframe(df.head())

# Show basic info
st.subheader("Dataset Information")

st.write("Jumlah Data:", df.shape[0])
st.write("Jumlah Kolom:", df.shape[1])

# Show columns
st.subheader("Columns")

st.write(df.columns.tolist())
