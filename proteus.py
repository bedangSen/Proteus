# -------------------------------------------------------------------------------------------------------------------
#  #---------------------------------------------------------------------------------------------------#            -
#  #  Proteus - Using Clustering Algorithms to identify Username Clusters in a Corporate Environment   #            -
#  #---------------------------------------------------------------------------------------------------#            -                                                                                       -
#                                                                                                                   -
#  Created by Bedang Sen                                                                                            -
#  Contact: Bedang.Sen@ibm.com | Sen.Bedang@protonmail.com                                                          -
#  Date: 11 September 2023                                                                                          -
#                                                                                                                   -
# -------------------------------------------------------------------------------------------------------------------

from argparse import ArgumentParser, HelpFormatter
from logging import basicConfig, getLogger, debug, error, info, warning
from os.path import basename, join, exists
from os import makedirs
from re import search, findall
from sys import exit
from pyfiglet import Figlet

# Importing pandas and numpy for data processing.
import pandas as pd
import numpy as np

# Importing scikit-learn modules for data preprocessing, clustering, and visualization.
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import plotly.express as px

# Importing rich modules for rich text and highlighted formatting.
from rich.logging import RichHandler
from rich.traceback import install
install()

basicConfig(
        level='INFO',
        format="%(message)s",
        handlers=[RichHandler(show_path=False, markup=True)]
    )
log = getLogger("proteus")

def parse_arguments():
    parser = ArgumentParser(
        prog="Proteus - Identifying Username Clusters in Corporate Environments",
        description="",
        epilog="IBM Security X-Force Incident Response",
        formatter_class=lambda prog: HelpFormatter(prog, max_help_position=52),
    )

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
    output_group.add_argument("--visualize", action="store_true", help="Visualize clusters", default=False)
    output_group.add_argument("--output_folder", default="output", help="Output folder")
    output_group.add_argument("--output_file", default="output.csv", help="Output file")
    
    logging_level = ["debug", "info", "warning", "error", "critical"]
    output_group.add_argument("--log", metavar="LOGGING_LEVEL", choices=logging_level, default="info", help=f"Select the logging level. Keep in mind increasing verbosity might affect performance. Available choices include : {logging_level}")

    args = parser.parse_args()
    init_logging(args.log)
    
    return args

def init_logging(log_level):
    basicConfig(
        level=log_level.upper(),
        format="%(message)s",
        handlers=[RichHandler()],
        # filename="myLog.log"
    )

    log = getLogger("proteus")
    log.setLevel(log_level.upper())

# Function to count the number of special characters in a username
def count_special_characters(username):
    special_characters = findall(r'[!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\\\]\^_`{\|}~]', username)
    return len(special_characters)

# Function to get a list of unique special characters in a username
def unique_special_characters(username):
    special_characters = findall(r'[!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\\\]\^_`{\|}~]', username)
    return list(set(special_characters))

def load_data(filename):
    try:
        # Load data from a CSV file and perform basic preprocessing
        log.info(f"[-] Loading data from {filename}...")
        df = pd.read_csv(filename, usecols=["Username"]).drop_duplicates()
        log.info(f" |__ [+] Loaded {len(df)} usernames from {filename}.")
        return df
    except Exception as e:
        log.error(f" |__ [!] Error loading data: {e}")
        exit(1)

def preprocess_data(df):
    # Implement your data preprocessing steps here
    # Feature Engineering, Scaling, etc.

    log.debug("[-] Performing feature engineering...")

    # Feature Engineering
    log.debug(" |__ [-] Extracting Feature: Length...");                          df['Length'] = df['Username'].apply(len)
    log.debug(" |__ [-] Extracting Feature: Special Characters...");              df['Special Characters'] = df['Username'].apply(lambda username: int(bool(search(r'[!\"#\$%&\'\(\)\*\+,\-\.\/:;<=>\?@\[\\\]\^_`{\|}~]', username))))
    log.debug(" |__ [-] Extracting Feature: Number of Special Characters...");    df['Number of Special Characters'] = df['Username'].apply(count_special_characters)
    log.debug(" |__ [-] Extracting Feature: Unique Special Characters...");       df['Unique Special Characters'] = df['Username'].apply(unique_special_characters) 
    log.debug(" |__ [-] Extracting Feature: Numbers...");                         df['Numbers'] = df['Username'].apply(lambda username: int(bool(search(r'\d', username))))
    log.debug(" |__ [-] Extracting Feature: Characters...");                      df['Characters'] = df['Username'].apply(lambda username: bool(search(r'[a-zA-Z]', username)))    
    log.debug(" |__ [-] Extracting Feature: Uppercase...");                       df['Uppercase'] = df['Username'].apply(lambda username: any(char.isupper() for char in username))
    log.debug(" |__ [-] Extracting Feature: Number of Words...");                 df['Number of Words'] = df['Username'].apply(lambda username: len(findall(r'\w+', username)))
    
    # Create binary columns for unique special characters
    log.debug(" |__ [-] Creating binary columns for uniue special characters...")
    unique_chars = df['Username'].apply(unique_special_characters)
    all_unique_chars = set(char for sublist in unique_chars for char in sublist)

    for char in all_unique_chars:
        df[char] = df['Username'].apply(lambda username: int(char in unique_special_characters(username)))

    # Define feature names
    log.debug(" |__ [-] Defining feature names...")
    all_feature_names = ['Uppercase', 'Characters', 'Special Characters', 'Numbers', 'Number of Words', 'Number of Special Characters'] + list(all_unique_chars)

    log.info(f" |__ [+] Performed feature engineering. {len(df)} usernames with {len(all_feature_names)} features.")
    
    return df, all_feature_names

def dbscan_clustering(df, eps, min_samples, all_feature_names):
    try:
        log.info("[-] Performing DBSCAN clustering...")

        # Perform DBSCAN clustering
        log.debug(" |__ [-] Scaling features...")
        scaler = StandardScaler()
        log.debug(" |__ [-] Fitting and transforming features...")
        scaled_features = scaler.fit_transform(df[all_feature_names])

        log.debug(" |__ [-] Performing DBSCAN clustering...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df['Cluster'] = dbscan.fit_predict(scaled_features)

        log.info(f" |__ [+] Performed DBSCAN clustering. {len(df['Cluster'].unique())} clusters found.")
        return df
    except Exception as e:
        log.error(f" |__ [!] Error performing DBSCAN clustering: {e}")
        exit(1)

def analyze_clusters(df, threshold, all_feature_names):
    log.info("[-] Analyzing clusters...")

    # Analyze clusters and contributing features
    log.debug(" |__ [-] Analyzing clusters and contributing features...")
    cluster_details_df = pd.DataFrame(columns=['Cluster', 'Username', 'Contributing Features'])
    log.debug(" |__ [-] Extracting unique clusters...")
    unique_clusters = df['Cluster'].unique()

    for cluster_label in unique_clusters:
        log.debug(f" |__ [-] Analyzing cluster {cluster_label}...")
        # if cluster_label == -1:
        #     log.debug(" |__ [-] Skipping noise cluster...")
        #     continue  # Skip noise cluster, if applicable

        # Perform cluster analysis and feature contribution here
        # Identify contributing features
        cluster_data = df[df['Cluster'] == cluster_label]
        feature_means = cluster_data[all_feature_names].mean()
        

        # Set a threshold to determine which features are considered as contributing significantly
        log.debug("      |__ [-] The following contributing features were identified:")
        contributing_features = feature_means[feature_means > threshold].index.tolist()
        log.debug(f"      |__ [-] {contributing_features}")
        

        # Store the contributing features for this cluster
        log.debug("      |__ [-] Storing contributing features...")
        cluster_details = {
            'Cluster': [cluster_label] * len(cluster_data),
            'Username': cluster_data['Username'].tolist(),
            'Contributing Features': [', '.join(contributing_features)] * len(cluster_data)
        }

        # Create a DataFrame for the current cluster and concatenate it with the cluster_details_df
        log.debug("      |__ [-] Creating DataFrame for the current cluster...")
        cluster_df = pd.DataFrame(cluster_details)
        cluster_details_df = pd.concat([cluster_details_df, cluster_df], ignore_index=True)

    log.info(f" |__ [+] Analyzed clusters. {len(cluster_details_df)} usernames analyzed.")

    return cluster_details_df

def visualize_clusters(df):
    try:
        log.info("[-] Visualizing clusters...")

        # Visualize clusters using t-SNE
        log.debug(" |__ [-] Performing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        scaled_features = df[all_feature_names].values
        reduced_features_tsne = tsne.fit_transform(scaled_features)

        log.debug(" |__ [-] Creating DataFrame for t-SNE...")
        reduced_df_tsne = pd.DataFrame(reduced_features_tsne, columns=['TSNE1', 'TSNE2'])
        reduced_df_tsne['Cluster'] = df['Cluster']

        # Create an interactive 2D scatter plot for t-SNE
        log.debug(" |__ [-] Creating interactive 2D scatter plot for t-SNE...")
        fig_tsne = px.scatter(
            reduced_df_tsne, x='TSNE1', y='TSNE2', color='Cluster',
            title='t-SNE 2D Scatter Plot of Clusters'
        )

        # Customize the t-SNE plot (optional)
        log.debug(" |__ [-] Customizing t-SNE plot...")
        fig_tsne.update_traces(marker=dict(size=5))
        fig_tsne.update_layout(legend_title_text='Cluster')
        fig_tsne.update_xaxes(title_text='TSNE1')
        fig_tsne.update_yaxes(title_text='TSNE2')

        # Show the interactive t-SNE plot in a web browser
        log.debug(" |__ [-] Showing interactive t-SNE plot...")
        fig_tsne.show()

        log.info(" |__ [+] Visualized clusters.")
        
    except Exception as e:
        log.error(f" |__ [!] Error visualizing clusters: {e}")

def output_data(df, args):
    # Output data to a CSV file based on the basename of the input file. Create it in the output folder provided by the user.
    log.info("[-] Outputting data...")

    # Create the output folder if it does not exist
    if not exists(args.output_folder):
        log.warning(f" |__ [!] Output folder {args.output_folder} does not exist.")
        log.info(f" |__ [-] Creating output folder {args.output_folder}...")
        makedirs(args.output_folder)
        log.info(f"      |__ [+] Created output folder {args.output_folder}.")

    df.to_csv(join(args.output_folder, args.output_file), index=False)
    log.info(f" |__ [+] Outputted data to {join(args.output_folder, args.output_file)}.")

def main(args):
    log.info(f"[bold green][+] {basename(__file__)} has started running successfully ...", extra={"markup": True})
    
    # Define feature names
    all_feature_names = ['Uppercase', 'Characters', 'Special Characters', 'Numbers', 'Number of Words', 'Number of Special Characters']
    
    df = load_data(args.filename)
    df, all_feature_names = preprocess_data(df)
    df = dbscan_clustering(df, args.eps, args.min_samples, all_feature_names)
    cluster_details_df = analyze_clusters(df, args.threshold, all_feature_names)
    output_data(cluster_details_df, args)

    if args.visualize:
        visualize_clusters(df)

    log.info(f"[bold green][+] {basename(__file__)} has completed running successfully ...", extra={"markup": True})


if __name__ == "__main__":
    fig = Figlet(font='colossal')
    print('\u001b[32;1m' + fig.renderText("Proteus") + '\033[39m')

    # Initialize arguments using arg parser
    args = parse_arguments()
    main(args)