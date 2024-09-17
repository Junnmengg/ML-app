import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Set page title and layout
st.title('Interactive Clustering and Anomaly Detection Dashboard')
st.sidebar.title('Options')

# Define the selected variables for clustering
selected_vars = [
    'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count',
    'num_compromised', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'total_bytes', 'serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# File upload
uploaded_files = st.sidebar.file_uploader("Upload your datasets", type=["csv", "xlsx"], accept_multiple_files=True)

# Check if two datasets are uploaded
if uploaded_files and len(uploaded_files) == 2:
    # Assume first dataset is MinMax scaled and second dataset is Standard scaled
    minmax_data = pd.read_csv(uploaded_files[0]) if uploaded_files[0].name.endswith('csv') else pd.read_excel(uploaded_files[0])
    standard_data = pd.read_csv(uploaded_files[1]) if uploaded_files[1].name.endswith('csv') else pd.read_excel(uploaded_files[1])

    # Select only the chosen columns
    X_minmax = minmax_data[selected_vars].dropna()
    X_standard = standard_data[selected_vars].dropna()

    st.write("Data Overviews:")
    st.write("MinMax Scaled Data Overview:")
    st.write(X_minmax.head())
    st.write("Standard Scaled Data Overview:")
    st.write(X_standard.head())
else:
    st.info("Please upload datasets")

# Function for interactive data exploration
def explore_data(data):
    st.subheader("Data Exploration")
    st.write(data)
    st.dataframe(data.describe())
    st.write("Select columns to visualize:")
    columns = st.multiselect("Columns", data.columns)
    if columns:
        fig = px.scatter_matrix(data[columns], title="Scatter Matrix Plot")
        st.plotly_chart(fig)

# Function to plot clusters
def plot_clusters(X, labels, title):
    fig = px.scatter(x=X[:, 0], y=X[:, 1], color=labels.astype(str), title=title, labels={'x': 'Feature 1', 'y': 'Feature 2'})
    st.plotly_chart(fig)

# Function to apply PCA after clustering
def apply_pca_after_clustering(data, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    plot_clusters(X_pca, labels, "PCA Visualization with Clusters")

# Function to evaluate clustering performance
def evaluate_clustering(X, labels):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    return silhouette, davies_bouldin, calinski_harabasz

# Initialize session state to store results
if 'comparative_results' not in st.session_state:
    st.session_state['comparative_results'] = pd.DataFrame(columns=['Model', 'Dataset', 'Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'])

# Select Clustering Algorithm
algorithm = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ["Gaussian Mixture Model (GMM)", "Hierarchical Clustering", "DBSCAN", "Spectral Clustering"]
)

if uploaded_files and len(uploaded_files) == 2:
    # Define clustering models
    def gmm_clustering(X):
        gmm = GaussianMixture(n_components=3, random_state=0)
        return gmm.fit_predict(X)

    def hdbscan_clustering(X):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
        return clusterer.fit_predict(X)

    def dbscan_clustering(X, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(X)

    def autoencoder_clustering(X):
        input_dim = X.shape[1]
        encoding_dim = 2
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=0)
        encoder = Model(input_layer, encoded)
        encoded_data = encoder.predict(X)
        return DBSCAN(eps=0.5, min_samples=5).fit_predict(encoded_data)

    def isolation_forest_clustering(X):
        iso_forest = IsolationForest(contamination=0.1)
        return iso_forest.fit_predict(X)

    # Sidebar options for DBSCAN
    if algorithm == "DBSCAN":
        eps = st.sidebar.slider('Epsilon (eps)', 0.1, 2.0, 0.5)
        min_samples = st.sidebar.slider('Min Samples', 1, 20, 5)
    else:
        eps = 0.5  # Default values
        min_samples = 5

    # Apply selected algorithm to both datasets
    for dataset_name, X in [("MinMax Scaled", X_minmax), ("Standard Scaled", X_standard)]:
        st.write(f"Processing {dataset_name} Dataset...")

        # Perform clustering
        if algorithm == "Gaussian Mixture Model (GMM)":
            labels = gmm_clustering(X)
        elif algorithm == "HDBSCAN":
            labels = hdbscan_clustering(X)
        elif algorithm == "DBSCAN":
            labels = dbscan_clustering(X, eps, min_samples)
        elif algorithm == "Autoencoder":
            labels = autoencoder_clustering(X)
        elif algorithm == "Isolation Forest":
            labels = isolation_forest_clustering(X)

        # Evaluate clustering performance
        silhouette, db, ch = evaluate_clustering(X, labels)

        # Update or add results to session state
        if not st.session_state['comparative_results'].empty:
            # Check if there is an existing entry for the same model and dataset
            existing_index = st.session_state['comparative_results'][
                (st.session_state['comparative_results']['Model'] == algorithm) &
                (st.session_state['comparative_results']['Dataset'] == dataset_name)
            ].index

            # If there is an existing entry, update it
            if len(existing_index) > 0:
                st.session_state['comparative_results'].loc[existing_index, ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']] = [silhouette, db, ch]
            else:
                # If no existing entry, append new result
                results_df = pd.DataFrame({
                    'Model': [algorithm],
                    'Dataset': [dataset_name],
                    'Silhouette Score': [silhouette],
                    'Davies-Bouldin Score': [db],
                    'Calinski-Harabasz Score': [ch]
                })
                st.session_state['comparative_results'] = pd.concat([st.session_state['comparative_results'], results_df], ignore_index=True)
        else:
            # If dataframe is empty, simply add the new results
            results_df = pd.DataFrame({
                'Model': [algorithm],
                'Dataset': [dataset_name],
                'Silhouette Score': [silhouette],
                'Davies-Bouldin Score': [db],
                'Calinski-Harabasz Score': [ch]
            })
            st.session_state['comparative_results'] = pd.concat([st.session_state['comparative_results'], results_df], ignore_index=True)

        # Display evaluation results
        st.write(f"**{dataset_name} Data**: Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {db:.4f}, Calinski-Harabasz Score: {ch:.4f}")

        # Apply PCA after clustering and plot
        st.subheader(f'PCA Visualization with Clustering for {dataset_name} Data')
        apply_pca_after_clustering(X, labels)

# Display all stored results
st.subheader("Comparative Analysis of Clustering Models")
st.dataframe(st.session_state['comparative_results'])

# Plot comparative analysis
fig = px.bar(st.session_state['comparative_results'],
             x='Model',
             y=['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score'],
             color='Dataset',
             barmode='group',
             title="Comparison of Clustering Models")
st.plotly_chart(fig)

# Button to update clustering in real-time
if st.sidebar.button('Update Clustering'):
    st.experimental_rerun()
