# Proteus

## Overview
This project aims to provide a tool for identifying naming convention clusters from a security perspective. By analyzing usernames, you can uncover patterns that may be indicative of security threats or user behaviors. The script allows you to perform hierarchical clustering on a dataset of usernames and discover clusters of similar naming conventions.

## Table of Contents
+ [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
+ [Usage](#usage)
    + [Input Data](#input-data)
    + [Running the Script](#running-the-script)
+ [Features](#features)
+ [Optimizing the Analysis](#optimizing-the-analysis)
+ [Demo](#demo)
+ [Contributing](#contributing)
+ [License](#license)

## Getting Started
### Prerequisites
To run this project, you'll need:

+ Python 3.x
+ Libraries mentioned in the requirements.txt file.

### Installation

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/your-username/username-clustering.git
    ```

1. Install the required Python libraries:

    ```
    pip install -r requirements.txt
    ```
## Usage
### Input Data
Before you can run the script, prepare your dataset of usernames in a CSV file with a header. The script assumes a column named "Username" containing the usernames to be analyzed.

### Running the Script

1. Navigate to the project directory:
    ```
    cd username-clustering
    ```

1. Run the script:

    ```
    python proteus.py
    ```
1. The script will perform the following steps:

    + Feature engineering to extract relevant information from usernames.
    + Hierarchical clustering to group similar naming conventions.
    + Silhouette analysis to determine the optimal number of clusters.
    + Output the clustered usernames and key features to a CSV file.


## Features
- [x] **Feature Engineering**: Extracts relevant features from usernames, including length, special characters, numbers, and more.

- [x] **Hierarchical Clustering**: Groups usernames into clusters based on similarity in naming conventions.

- [x] **Silhouette Analysis**: Determines the optimal number of clusters for the dataset.

- [x] **Key Feature Identification**: Identifies key features that define each cluster.

- [x] **Visualization**: Visualizes the hierarchical clustering using dendrograms and silhouette score plots.

## Optimizing the Analysis
To enhance the effectiveness of the username clustering analysis:

+ Experiment with different clustering algorithms and parameters.
+ Explore additional features that may reveal naming conventions.
+ Continuously gather feedback from security experts to refine the analysis.

## Demo
Check out the project in action!

Watch Demo Video

Demo GIF

## Contributing
Contributions are welcome!

## License
This project is licensed under the MIT License - see the LICENSE file for details.