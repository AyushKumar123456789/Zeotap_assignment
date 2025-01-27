
"""
Task 3: Customer Segmentation (Clustering)

This script:
1. Reads Customer, Transaction, and Product data.
2. Engineers features for each customer.
3. Applies K-Means clustering with optimal K determined by Davies-Bouldin Index.
4. Calculates clustering metrics, including the DB Index.
5. Visualizes clusters using PCA.
6. Generates a comprehensive PDF report with clustering results and visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
import sys

def load_and_merge_data(customers_path, products_path, transactions_path):
    """
    Loads and merges the Customers, Products, and Transactions datasets.

    Args:
        customers_path (str): Path to Customers.csv
        products_path (str): Path to Products.csv
        transactions_path (str): Path to Transactions.csv

    Returns:
        pd.DataFrame: Merged dataframe containing all necessary information.
        pd.DataFrame: Customers dataframe.
    """
    try:
        customers_df = pd.read_csv(customers_path)
        products_df = pd.read_csv(products_path)
        transactions_df = pd.read_csv(transactions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Drop duplicates
    customers_df.drop_duplicates(inplace=True)
    products_df.drop_duplicates(inplace=True)
    transactions_df.drop_duplicates(inplace=True)

    # Convert date columns to datetime
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'], errors='coerce')
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')

    # Handle missing dates by filling with median date
    median_signup = customers_df['SignupDate'].median()
    median_transaction = transactions_df['TransactionDate'].median()
    customers_df['SignupDate'].fillna(median_signup, inplace=True)
    transactions_df['TransactionDate'].fillna(median_transaction, inplace=True)

    # Merge dataframes
    merged_df = pd.merge(transactions_df, customers_df, on='CustomerID', how='left')
    merged_df = pd.merge(merged_df, products_df, on='ProductID', how='left')

    return merged_df, customers_df

def engineer_features(merged_df, customers_df):
    """
    Engineers features required for clustering.

    Args:
        merged_df (pd.DataFrame): Merged dataframe.
        customers_df (pd.DataFrame): Customers dataframe.

    Returns:
        pd.DataFrame: Feature dataframe ready for clustering.
    """
    # Current date for calculating DaysSinceSignup
    current_date = merged_df['TransactionDate'].max()

    # TotalSpent and NumTransactions
    customer_spending = merged_df.groupby('CustomerID')['TotalValue'].agg(['sum', 'count']).reset_index()
    customer_spending.columns = ['CustomerID', 'TotalSpent', 'NumTransactions']
    customer_spending['AvgOrderValue'] = customer_spending['TotalSpent'] / customer_spending['NumTransactions']

    # CategorySpend
    category_spend = merged_df.groupby(['CustomerID', 'Category'])['TotalValue'].sum().unstack(fill_value=0)
    category_spend.reset_index(inplace=True)

    # DaysSinceSignup
    customers_df['DaysSinceSignup'] = (current_date - customers_df['SignupDate']).dt.days

    # Merge features
    feature_df = pd.merge(customers_df[['CustomerID', 'Region', 'DaysSinceSignup']], 
                          customer_spending, 
                          on='CustomerID', how='left')

    feature_df = pd.merge(feature_df, category_spend, on='CustomerID', how='left')
    feature_df.fillna(0, inplace=True)

    # Convert categorical 'Region' into dummy variables
    feature_df = pd.get_dummies(feature_df, columns=['Region'], prefix='Region')

    return feature_df

def scale_features(feature_df):
    """
    Scales the feature dataframe using Min-Max Scaler.

    Args:
        feature_df (pd.DataFrame): Feature dataframe.

    Returns:
        np.ndarray: Scaled feature array.
        MinMaxScaler: Fitted scaler object.
    """
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(feature_df)
    return features_scaled, scaler

def determine_optimal_k(features_scaled, k_min=2, k_max=10):
    """
    Determines the optimal number of clusters (K) using Davies-Bouldin Index.

    Args:
        features_scaled (np.ndarray): Scaled feature array.
        k_min (int): Minimum number of clusters to try.
        k_max (int): Maximum number of clusters to try.

    Returns:
        int: Optimal number of clusters.
        dict: Dictionary containing DB Index for each K.
    """
    db_index_scores = {}
    silhouette_scores = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        db_index = davies_bouldin_score(features_scaled, cluster_labels)
        silhouette = silhouette_score(features_scaled, cluster_labels)
        db_index_scores[k] = db_index
        silhouette_scores[k] = silhouette
        print(f"K={k}: DB Index={db_index:.4f}, Silhouette Score={silhouette:.4f}")

    # Select K with the lowest DB Index
    optimal_k = min(db_index_scores, key=db_index_scores.get)
    print(f"\nOptimal number of clusters based on lowest DB Index: K={optimal_k}")

    return optimal_k, db_index_scores

def apply_clustering(features_scaled, k):
    """
    Applies K-Means clustering to the scaled features.

    Args:
        features_scaled (np.ndarray): Scaled feature array.
        k (int): Number of clusters.

    Returns:
        np.ndarray: Cluster labels.
        KMeans: Fitted KMeans object.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return cluster_labels, kmeans

def visualize_clusters_pca(features_scaled, cluster_labels, output_path):
    """
    Visualizes clusters using PCA for dimensionality reduction.

    Args:
        features_scaled (np.ndarray): Scaled feature array.
        cluster_labels (np.ndarray): Cluster labels.
        output_path (str): Path to save the cluster plot.
    """
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(data=components, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = cluster_labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set2', s=100, alpha=0.7)
    plt.title('Customer Segments Visualized with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster', loc='best')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Cluster visualization saved as {output_path}")

def generate_pdf_report(optimal_k, db_index, silhouette, cluster_sizes, output_path, cluster_plot_path):
    """
    Generates a PDF report containing clustering results and visualizations.

    Args:
        optimal_k (int): Number of clusters.
        db_index (float): Davies-Bouldin Index.
        silhouette (float): Silhouette Score.
        cluster_sizes (dict): Dictionary with cluster labels as keys and sizes as values.
        output_path (str): Path to save the PDF report.
        cluster_plot_path (str): Path to the cluster plot image.
    """
    class PDFReport(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'Customer Segmentation Report', 0, 1, 'C')

        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.multi_cell(0, 10, title, 0, 'L')
            self.ln(2)

        def chapter_body(self, body):
            self.set_font('Arial', '', 12)
            self.multi_cell(0, 10, body)
            self.ln()

        def add_image(self, image_path, title):
            if os.path.exists(image_path):
                self.chapter_title(title)
                self.image(image_path, w=180)
                self.ln(10)
            else:
                self.chapter_title(title)
                self.chapter_body("Image not found.")
    
    pdf = PDFReport()
    pdf.add_page()

    # Clustering Summary
    pdf.chapter_title('Clustering Summary')
    summary = f"Optimal Number of Clusters (K): {optimal_k}\n"
    summary += f"Davies-Bouldin Index: {db_index:.4f}\n"
    summary += f"Silhouette Score: {silhouette:.4f}\n\n"
    summary += "Cluster Sizes: \n"
    for cluster, size in cluster_sizes.items():
        summary += f"- Cluster {cluster}: {size} customers\n"
    pdf.chapter_body(summary)

    # Clustering Metrics
    pdf.chapter_title('Clustering Metrics')
    metrics = (
        f"The Davies-Bouldin Index (DB Index) for K={optimal_k} is {db_index:.4f}. "
        f"A lower DB Index indicates better clustering with well-separated clusters.\n\n"
        f"The Silhouette Score for K={optimal_k} is {silhouette:.4f}. "
        f"A higher Silhouette Score indicates better-defined clusters.\n"
    )
    pdf.chapter_body(metrics)

    # Cluster Visualization
    pdf.add_image(cluster_plot_path, 'Cluster Visualization with PCA')

    # Save the PDF
    pdf.output(output_path)
    print(f"Clustering PDF report generated successfully at {output_path}!")

def main():
    """
    Main function to execute the Customer Segmentation tasks.
    """
    # Define paths
    customers_path = "../data/Customers.csv"
    products_path = "../data/Products.csv"
    transactions_path = "../data/Transactions.csv"
    clustering_pdf_path = "Ayush_Kumar_Clustering.pdf"
    cluster_plot_path = "ClusterPlot.png"

    # Step 1: Load and merge data
    merged_df, customers_df = load_and_merge_data(customers_path, products_path, transactions_path)

    # Step 2: Engineer features
    feature_df = engineer_features(merged_df, customers_df)

    # Step 3: Scale features
    features_scaled, scaler = scale_features(feature_df.drop('CustomerID', axis=1))

    # Step 4: Determine optimal K
    optimal_k, db_index_scores = determine_optimal_k(features_scaled, k_min=2, k_max=10)
    db_index = db_index_scores[optimal_k]

    # Additionally, calculate silhouette score for the optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    silhouette = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette Score for K={optimal_k}: {silhouette:.4f}")

    # Step 5: Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Step 6: Visualize clusters
    visualize_clusters_pca(features_scaled, cluster_labels, cluster_plot_path)

    # Step 7: Generate PDF report
    generate_pdf_report(optimal_k, db_index, silhouette, cluster_sizes, clustering_pdf_path, cluster_plot_path)

if __name__ == "__main__":
    main()
