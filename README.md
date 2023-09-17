# Proteus

## Overview

Proteus is a tool designed to identify naming convention clusters from a security perspective. By analyzing usernames, it helps you uncover patterns indicative of security threats or user behaviors. The script performs hierarchical clustering on a dataset of usernames to discover clusters with similar naming conventions.

https://github.com/bedangSen/Proteus/assets/35425072/8de166b2-e9cd-495a-9a77-9e367f0f39e5

## Table of Contents
+ [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
+ [Usage](#usage)
    + [Input Data](#input-data)
    + [Running the Script](#running-the-script)
    + [Visualizing Clusters](#visualizing-clusters)
+ [Features](#features)
+ [Customization](#customization)
+ [Demo](#demo)
+ [Contributing](#contributing)
+ [License](#license)

## Getting Started

### Prerequisites
To run Proteus, ensure you have the following prerequisites installed:

+ Python 3.x
+ Required libraries mentioned in the requirements.txt file.

### Installation
1. Clone this repository to your local machine:
    ```
    git clone https://github.com/your-username/Proteus.git
    ```

1. Install the necessary Python libraries:

    ```
    pip install -r requirements.txt
    ```

## Usage

### Input Data
Before running the script, prepare your dataset of usernames in a CSV file with a header. The script expects a column named "Username" containing the usernames for analysis.

### Running the Script
1. Navigate to the project directory:
    ```
    cd Proteus
    ```

1. Execute the script:
    ```
    python proteus.py --filename your_data.csv
    ```

### Visualizing Clusters
You can visualize the clusters by adding the --visualise_clusters flag when running the script. This will generate interactive t-SNE plots to help you explore the clustering results.
    ```
    python proteus.py --filename your_data.csv --visualise_clusters
    ```

## Features
- [x] **Feature Engineering**: Extracts relevant features from usernames, including length, special characters, numbers, and more.

- [x] **Hierarchical Clustering**: Groups usernames into clusters based on similarity in naming conventions.

- [x] **Key Feature Identification**: Identifies key features that define each cluster.

- [x] **Visualization**: Visualizes the hierarchical clustering using dendrograms and silhouette score plots.

## Customisation

You can customize Proteus to tailor its behavior to your specific dataset and analysis requirements by adjusting the following parameters:

+ `--eps`: The DBSCAN (Density-Based Spatial Clustering of Applications with Noise) epsilon parameter controls the maximum distance between two samples for one to be considered as in the neighborhood of the other. In simpler terms, it defines how close points need to be to form a cluster. A smaller value will result in tighter, more compact clusters, while a larger value may merge clusters or classify points as outliers. Experiment with different epsilon values to find the optimal one for your data.

+ `--min_samples`: This parameter sets the minimum number of samples (data points) required to form a cluster. It determines the minimum cluster size. Increasing this value can make the algorithm more conservative, resulting in fewer and larger clusters. Decreasing it can lead to more clusters and potentially smaller, more fine-grained clusters. Choose a value that aligns with the size and distribution of your data.

+ `--threshold`: The threshold for feature contribution controls which features are considered significant in defining clusters. Features with mean values above this threshold are considered as contributing features for a cluster. By adjusting this threshold, you can fine-tune the sensitivity of the analysis to feature contributions. Lowering the threshold may result in more features being considered, potentially leading to larger and more complex clusters. Raising it can result in more focused and distinctive clusters.

+ `--output_folder`: This parameter specifies the folder where Proteus will save the output files, including the clustered usernames and key features. By setting this folder, you can organize and store the results in a location that suits your project's structure.

+ `--output_file`: Use this parameter to set the name of the output CSV file where the clustered usernames and key features will be saved. You can customize the file name to match your project's naming conventions.

+ `--visualise_clusters`: Enabling this parameter (--visualise_clusters) triggers the generation of interactive t-SNE (t-Distributed Stochastic Neighbor Embedding) plots. These plots provide a visual representation of the clustering results. t-SNE is a dimensionality reduction technique that can help you explore the structure of your clusters in a 2D space. This feature is particularly useful for gaining insights into how your usernames are grouped.

## Demo
See Proteus in action! Demo Video (Coming Soon)

## Contributing
Contributions to Proteus are welcome. Feel free to open issues, suggest enhancements, or submit pull requests.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
