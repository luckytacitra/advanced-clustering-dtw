import streamlit as st
import base64
import pandas as pd
import plotly.express as px
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =====================================
# PAGE CONFIG
# =====================================
# =====================================
# LOAD SIDEBAR BACKGROUND
# =====================================

def get_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()

    return base64.b64encode(data).decode()

bg_image = get_base64("bg.jpg")

st.set_page_config(
    page_title="Advanced DTW Clustering",
    page_icon="🚗",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown(
    f'''
    <style>

    .main {{
        background-color: #f8fbff;
    }}

    h1, h2, h3 {{
        color: #1e3a5f;
        font-family: 'Trebuchet MS';
    }}

    section[data-testid="stSidebar"] {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
    }}

    section[data-testid="stSidebar"] > div {{
        background: rgba(255,255,255,0.55);
        backdrop-filter: blur(10px);
    }}

    .stMetric {{
        background: rgba(255,255,255,0.75);
        border-radius: 18px;
        padding: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }}

    div[data-testid="stDataFrame"] {{
        background: rgba(255,255,255,0.8);
        border-radius: 15px;
        padding: 10px;
    }}

    .stPlotlyChart {{
        background: rgba(255,255,255,0.75);
        border-radius: 20px;
        padding: 10px;
    }}

    </style>
    ''',
    unsafe_allow_html=True
)

# =====================================
# TITLE
# =====================================
st.title("🚗 Advanced Time Series Clustering with DTW")

st.markdown(
    """
    Dashboard analisis clustering time series menggunakan:
    - Dynamic Time Warping (DTW)
    - TimeSeriesKMeans
    - PCA Visualization
    - Silhouette Score
    """
)

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("dataset.csv")

# =====================================
# PREPROCESSING
# =====================================
df['LastUpdated'] = pd.to_datetime(df['LastUpdated'])

# Occupancy Rate
# occupancy / capacity

df['OccupancyRate'] = (
    df['Occupancy'] / df['Capacity']
)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("⚙️ Configuration")

n_clusters = st.sidebar.slider(
    "Select Number of Clusters",
    min_value=2,
    max_value=6,
    value=3
)

selected_parking = st.sidebar.selectbox(
    "Choose Parking Area",
    df['SystemCodeNumber'].unique()
)

# =====================================
# DATASET OVERVIEW
# =====================================
st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Jumlah Data",
        df.shape[0]
    )

with col2:
    st.metric(
        "Jumlah Kolom",
        df.shape[1]
    )

with col3:
    st.metric(
        "Jumlah Parkiran",
        df['SystemCodeNumber'].nunique()
    )

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =====================================
# OCCUPANCY RATE PREVIEW
# =====================================
st.subheader("Occupancy Rate Preview")

st.dataframe(
    df[[
        'SystemCodeNumber',
        'Occupancy',
        'Capacity',
        'OccupancyRate'
    ]].head()
)

# =====================================
# TIME SERIES VISUALIZATION
# =====================================
st.header("📈 Time Series Visualization")

filtered_df = df[
    df['SystemCodeNumber'] == selected_parking
]

fig_ts = px.line(
    filtered_df,
    x='LastUpdated',
    y='OccupancyRate',
    title=f'Occupancy Rate - {selected_parking}',
    template='plotly_white'
)

fig_ts.update_traces(
    line_color='#5DADE2'
)

st.plotly_chart(
    fig_ts,
    use_container_width=True
)

# =====================================
# TIME SERIES PIVOT
# =====================================
st.header("🧠 DTW Time Series Clustering")

pivot_df = df.pivot_table(
    index='LastUpdated',
    columns='SystemCodeNumber',
    values='OccupancyRate'
)

# Fill missing values
pivot_df = pivot_df.ffill().bfill()

# =====================================
# SCALING
# =====================================
data = pivot_df.T.values

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

# =====================================
# TIME SERIES FORMAT
# =====================================
ts_data = to_time_series_dataset(data_scaled)

# =====================================
# DTW CLUSTERING
# =====================================
model = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric='dtw',
    random_state=42
)

labels = model.fit_predict(ts_data)

# =====================================
# CLUSTER RESULT
# =====================================
cluster_result = pd.DataFrame({
    'ParkingLot': pivot_df.columns,
    'Cluster': labels
})

st.subheader("Cluster Results")

st.dataframe(cluster_result)

# =====================================
# SILHOUETTE SCORE
# =====================================
score = silhouette_score(
    data_scaled,
    labels
)

st.metric(
    "Silhouette Score",
    round(score, 4)
)

# =====================================
# CLUSTER DISTRIBUTION
# =====================================
st.subheader("Cluster Distribution")

cluster_count = cluster_result['Cluster'].value_counts()

fig_bar = px.bar(
    x=cluster_count.index,
    y=cluster_count.values,
    labels={
        'x': 'Cluster',
        'y': 'Number of Parking Lots'
    },
    title='Distribution of Clusters',
    template='plotly_white'
)

fig_bar.update_traces(
    marker_color='#85C1E9'
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# =====================================
# PCA VISUALIZATION
# =====================================
st.header("🌌 PCA Cluster Visualization")

pca = PCA(n_components=2)

pca_result = pca.fit_transform(data_scaled)

pca_df = pd.DataFrame({
    'PCA1': pca_result[:, 0],
    'PCA2': pca_result[:, 1],
    'Cluster': labels.astype(str),
    'ParkingLot': pivot_df.columns
})

fig_cluster = px.scatter(
    pca_df,
    x='PCA1',
    y='PCA2',
    color='Cluster',
    hover_name='ParkingLot',
    title='DTW Time Series Clustering',
    template='plotly_white',
    color_discrete_sequence=[
    '#007BFF',  # biru
    '#00C9A7',  # tosca
    '#FFD93D',  # kuning
    '#FF6B6B',  # merah coral
    '#9B59B6',  # ungu
    '#FF8C42'   # orange
]
)

st.plotly_chart(
    fig_cluster,
    use_container_width=True
)

# =====================================
# DOWNLOAD RESULT
# =====================================
st.header("⬇️ Download Cluster Result")

csv = cluster_result.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name='cluster_result.csv',
    mime='text/csv'
)

# =====================================
# INTERPRETATION
# =====================================
st.header("📝 Cluster Interpretation")

st.markdown(
    """
    ### Interpretasi Hasil

    - Cluster yang sama menunjukkan pola okupansi parkir yang mirip.
    - DTW mampu mendeteksi kemiripan pola meskipun terdapat pergeseran waktu.
    - Clustering dilakukan menggunakan TimeSeriesKMeans dengan DTW distance.
    - PCA digunakan untuk memvisualisasikan hasil clustering ke dalam 2 dimensi.
    """
)
