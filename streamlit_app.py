import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Set page title and layout
st.title('Interactive Clustering Dashboard')
st.sidebar.title('Options')

# Define the selected variables for clustering
selected_vars = [
    'Annual_Income', 'Kidhome', 'Teenhome', 'Recency',
    'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

# Check if the dataset is uploaded
if uploaded_file:
    
    marketing_campaign_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)

    # Select only the chosen columns
    X_marketing_campaign = marketing_campaign_data[selected_vars].dropna()

    st.write("Data Overview:")
    st.write(X_marketing_campaign.head())
else:
    st.info("Please upload a dataset")

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
def apply_pca_after_clustering(data, labels, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data)
    plot_clusters(X_pca, labels, f"PCA Visualization with {n_components} Components and Clusters")

# Function to evaluate clustering performance
def evaluate_clustering(X, labels):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    return silhouette, davies_bouldin, calinski_harabasz

# Select Clustering Algorithm
algorithm = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ["Gaussian Mixture Model (GMM)", "Hierarchical Clustering", "DBSCAN", "Spectral Clustering"]
)

# User input for number of clusters
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

# User input for number of PCA components
n_pca_components = st.sidebar.slider("Number of PCA Components", 2, 10, 2)

if uploaded_file:
    # Define clustering models
    def gmm_clustering(X, n_clusters):
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        return labels, bic, aic

    def hierarchical_clustering(X, n_clusters):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return labels

    def dbscan_clustering(X, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(X)

    def spectral_clustering(X, n_clusters):
        model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
        return model.fit_predict(X)

    # Sidebar options for DBSCAN
    if algorithm == "DBSCAN":
        eps = st.sidebar.slider('Epsilon (eps)', 0.1, 2.0, 0.5)
        min_samples = st.sidebar.slider('Min Samples', 1, 20, 5)
    else:
        eps = 0.5  # Default values
        min_samples = 5

    # Perform clustering
    st.write(f"Processing data with {algorithm}...")

    if algorithm == "Gaussian Mixture Model (GMM)":
        labels, bic, aic = gmm_clustering(X_marketing_campaign, n_clusters)
        st.write(f"BIC Score: {bic:.4f}, AIC Score: {aic:.4f}")
    elif algorithm == "Hierarchical Clustering":
        labels = hierarchical_clustering(X_marketing_campaign, n_clusters)
        # Plot dendrogram for hierarchical clustering
        st.subheader("Dendrogram for Hierarchical Clustering")
        Z = linkage(X_marketing_campaign, 'ward')
        fig, ax = plt.subplots()
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=n_clusters)
        st.pyplot(fig)
    elif algorithm == "DBSCAN":
        labels = dbscan_clustering(X_marketing_campaign, eps, min_samples)
    elif algorithm == "Spectral Clustering":
        labels = spectral_clustering(X_marketing_campaign, n_clusters)

    # Evaluate clustering performance
    silhouette, db, ch = evaluate_clustering(X_marketing_campaign, labels)

    # Display evaluation results
    st.write(f"Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {db:.4f}, Calinski-Harabasz Score: {ch:.4f}")

    # Apply PCA after clustering and plot
    st.subheader(f'PCA Visualization with Clustering')
    apply_pca_after_clustering(X_marketing_campaign, labels, n_pca_components)

# Button to update clustering in real-time
if st.sidebar.button('Update Clustering'):
    st.experimental_rerun()
