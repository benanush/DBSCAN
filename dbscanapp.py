import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Wine DBSCAN Clustering",
    layout="centered"
)

st.title("Dataset Clustering using DBSCAN")

# --------------------------------------------------
# Step 1: Load Dataset
# --------------------------------------------------
st.header("📂 Step 1: Load Dataset")

uploaded_file = st.file_uploader(
    "upload a csv file ",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Step 2: Data Overview
    # --------------------------------------------------
    st.header("🔍 Step 2: Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Shape**")
        st.write(df.shape)

    with col2:
        st.write("**Missing Values**")
        st.write(df.isnull().sum())

    # --------------------------------------------------
    # Step 3: Feature Distributions
    # --------------------------------------------------
    st.header("📊 Step 3: Feature Distributions")

    if st.checkbox("Show Histograms"):
        fig, ax = plt.subplots(figsize=(10, 8))
        df.hist(bins=20, edgecolor='black', ax=ax)
        plt.suptitle("Feature Distribution")
        st.pyplot(fig)

    # --------------------------------------------------
    # Step 4: Standardization
    # --------------------------------------------------
    st.header("⚙️ Step 4: Standardization")

    numerical_features = df.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_features)
    df_scaled = pd.DataFrame(scaled_data, columns=numerical_features.columns)

    st.write("Standardized Data Preview")
    st.dataframe(df_scaled.head())

    # --------------------------------------------------
    # Step 5: DBSCAN Parameters
    # --------------------------------------------------
    st.header("🔧 Step 5: DBSCAN Parameters")

    eps = st.slider("Select eps", 0.5, 5.0, 2.0, 0.1)
    min_samples = st.slider("Select min_samples", 1, 10, 2)

    # --------------------------------------------------
    # Step 6: Apply DBSCAN
    # --------------------------------------------------
    st.header("🧠 Step 6: Apply DBSCAN")

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_scaled)

    df['Cluster'] = clusters

    st.write("Clustered Data Preview")
    st.dataframe(df.head())

    st.write("Cluster Distribution")
    st.bar_chart(df['Cluster'].value_counts())

    # --------------------------------------------------
    # Step 7: Cluster Visualization
    # --------------------------------------------------
    st.header("📈 Step 7: Cluster Visualization")

    x_feature = st.selectbox("Select X-axis feature", numerical_features.columns)
    y_feature = st.selectbox("Select Y-axis feature", numerical_features.columns)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df[x_feature],
        df[y_feature],
        c=df['Cluster'],
        cmap='viridis',
        s=60
    )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title("DBSCAN Clustering Visualization")
    plt.colorbar(scatter)
    st.pyplot(fig)

else:
    st.info("👆 Please upload a csv file to continue.")
