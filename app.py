import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("🚗 Advanced Time Series Clustering with DTW")

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert datetime
df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])

# Create Occupancy Rate
df['OccupancyRate'] = df['Occupancy'] / df['Capacity']

# Show dataset
st.subheader("Dataset Preview")

st.dataframe(df.head())

st.subheader("Occupancy Rate Preview")

st.dataframe(
    df[['SystemCodeNumber',
        'Occupancy',
        'Capacity',
        'OccupancyRate']]
    .head()
)

# Show basic info
st.subheader("Dataset Information")

st.write("Jumlah Data:", df.shape[0])
st.write("Jumlah Kolom:", df.shape[1])

# Show columns
st.subheader("Columns")

st.write(df.columns.tolist())

# Select parking lot
st.subheader("Parking Occupancy Time Series")

parking_choice = st.selectbox(
    "Choose Parking Area",
    df['SystemCodeNumber'].unique()
)

# Filter data
filtered_df = df[
    df['SystemCodeNumber'] == parking_choice
]

# Plot time series
fig = px.line(
    filtered_df,
    x='LastUpdated',
    y='OccupancyRate',
    title=f"Occupancy Rate - {parking_choice}"
)

st.plotly_chart(fig)
