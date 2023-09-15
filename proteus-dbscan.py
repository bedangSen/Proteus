import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import plotly.express as px
import argparse
import sys

# Function to count the number of special characters in a username
def count_special_characters(username):
    special_characters = re.findall(r'[!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\\\]\^_`{\|}~]', username)
    return len(special_characters)

# Function to get a list of unique special characters in a username
def unique_special_characters(username):
    special_characters = re.findall(r'[!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\\\]\^_`{\|}~]', username)
    return list(set(special_characters))

def load_data(filename):
    try:
        # Load data from a CSV file and perform basic preprocessing
        df = pd.read_csv(filename, usecols=["Username"]).drop_duplicates()
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df):
    # Implement your data preprocessing steps here
    # Feature Engineering, Scaling, etc.

    # Feature Engineering
    df['Length'] = df['Username'].apply(len)
    df['Special Characters'] = df['Username'].apply(lambda username: int(bool(re.search(r'[!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\\\]\^_`{\|}~]', username))))
    df['Number of Special Characters'] = df['Username'].apply(count_special_characters)
    df['Unique Special Characters'] = df['Username'].apply(unique_special_characters)
    df['Numbers'] = df['Username'].apply(lambda username: int(bool(re.search(r'\d', username))))
    df['Characters'] = df['Username'].apply(lambda username: bool(re.search(r'[a-zA-Z]', username)))
    df['Uppercase'] = df['Username'].apply(lambda username: any(char.isupper() for char in username))
    df['Number of Words'] = df['Username'].apply(lambda username: len(re.findall(r'\w+', username)))

    # Create binary columns for unique special characters
    unique_chars = df['Username'].apply(unique_special_characters)
    all_unique_chars = set(char for sublist in unique_chars for char in sublist)

    for char in all_unique_chars:
        df[char] = df['Username'].apply(lambda username: int(char in unique_special_characters(username)))

    # Define feature names
    all_feature_names = ['Uppercase', 'Characters', 'Special Characters', 'Numbers', 'Number of Words', 'Number of Special Characters'] + list(all_unique_chars)

    return df, all_feature_names

def dbscan_clustering(df, eps, min_samples, all_feature_names):
    try:
        # Perform DBSCAN clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[all_feature_names])

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = dbscan.fit_predict(scaled_features)
        return df
    except Exception as e:
        print(f"Error performing DBSCAN clustering: {e}")
        sys.exit(1)

def analyze_clusters(df, threshold):
    # Analyze clusters and contributing features
    cluster_details_df = pd.DataFrame(columns=['Cluster', 'Username', 'Contributing Features'])
    unique_clusters = df['Cluster'].unique()

    for cluster_label in unique_clusters:
        if cluster_label == -1:
            continue  # Skip noise cluster, if applicable

        # Perform cluster analysis and feature contribution here
        # Identify contributing features
        cluster_data = df[df['Cluster'] == cluster_label]
        feature_means = cluster_data[all_feature_names].mean()

        # Set a threshold to determine which features are considered as contributing significantly
        contributing_features = feature_means[feature_means > threshold].index.tolist()

        # Store the contributing features for this cluster
        cluster_details = {
            'Cluster': [cluster_label] * len(cluster_data),
            'Username': cluster_data['Username'].tolist(),
            'Contributing Features': [', '.join(contributing_features)] * len(cluster_data)
        }

        # Create a DataFrame for the current cluster and concatenate it with the cluster_details_df
        cluster_df = pd.DataFrame(cluster_details)
        cluster_details_df = pd.concat([cluster_details_df, cluster_df], ignore_index=True)

    return cluster_details_df

def visualize_clusters(df):
    try:
        # Visualize clusters using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        scaled_features = df[all_feature_names].values
        reduced_features_tsne = tsne.fit_transform(scaled_features)

        reduced_df_tsne = pd.DataFrame(reduced_features_tsne, columns=['TSNE1', 'TSNE2'])
        reduced_df_tsne['Cluster'] = df['Cluster']

        # Create an interactive 2D scatter plot for t-SNE
        fig_tsne = px.scatter(
            reduced_df_tsne, x='TSNE1', y='TSNE2', color='Cluster',
            title='t-SNE 2D Scatter Plot of Clusters'
        )

        # Customize the t-SNE plot (optional)
        fig_tsne.update_traces(marker=dict(size=5))
        fig_tsne.update_layout(legend_title_text='Cluster')
        fig_tsne.update_xaxes(title_text='TSNE1')
        fig_tsne.update_yaxes(title_text='TSNE2')

        # Show the interactive t-SNE plot in a web browser
        fig_tsne.show()
        
    except Exception as e:
        print(f"Error visualizing clusters: {e}")

def output_data(df, args):
    # Output data to a CSV file based on the basename of the input file. Create it in the output folder provided by the user.

    # Create the output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    df.to_csv(os.path.join(args.output_folder, args.output_file), index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Username Clustering and Analysis")

    # Create a parser group for inputs.
    input_group = parser.add_argument_group("Inputs")
    input_group.add_argument("--filename", required=True, help="CSV file containing usernames")

    # Create a parser group for configuring the parameters of DBSCAN.
    dbscan_group = parser.add_argument_group("DBSCAN Parameters")
    dbscan_group.add_argument("--eps", type=float, default=0.5, help="DBSCAN epsilon parameter")
    dbscan_group.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples parameter")
    dbscan_group.add_argument("--threshold", type=float, default=0.5, help="Threshold for feature contribution")

    # Create a parser group for output options.
    output_group = parser.add_argument_group("Outputs")
    output_group.add_argument("--output_folder", default="output", help="Output folder")
    output_group.add_argument("--output_file", default="output.csv", help="Output file")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Define feature names
    all_feature_names = ['Uppercase', 'Characters', 'Special Characters', 'Numbers', 'Number of Words', 'Number of Special Characters']

    args = parse_arguments()
    
    df = load_data(args.filename)
    df, all_feature_names = preprocess_data(df)
    df = dbscan_clustering(df, args.eps, args.min_samples, all_feature_names)
    cluster_details_df = analyze_clusters(df, args.threshold)
    output_data(cluster_details_df, args)
    visualize_clusters(df)