# Function Call Graph Processing Pipeline

This repository contains a pipeline for processing function call graphs (FCGs), training a Word2Vec model, preprocessing graphs, and training a Graph Convolutional Network (GCN) with attention. The pipeline is divided into four stages, each performing a specific task.

## Table of Contents

- [Function Call Graph Processing Pipeline](#function-call-graph-processing-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Clone the repository:](#clone-the-repository)
    - [Create and activate a virtual environment:](#create-and-activate-a-virtual-environment)
    - [Install the required packages:](#install-the-required-packages)
  - [Usage](#usage)
  - [Pipeline Stages](#pipeline-stages)
    - [Stage 1: Normalize Function Call Graphs](#stage-1-normalize-function-call-graphs)
    - [Stage 2: Train Word2Vec Model](#stage-2-train-word2vec-model)
    - [Stage 3: Preprocess Graphs](#stage-3-preprocess-graphs)
    - [Stage 4: Train GCN Model](#stage-4-train-gcn-model)
  - [Configuration](#configuration)
  - [License](#license)

## Prerequisites

* Python 3.7 or later
* PyTorch
* Torch Geometric
* Other dependencies listed in `requirements.txt`

## Installation

### Clone the repository:

```bash
git clone https://github.com/YuCheng1122/Embedding-vector-generation-based-on-function-call-graph-.git
cd Embedding-vector-generation-based-on-function-call-graph
```
### Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### Install the required packages:
```bash
pip install -r requirements.txt
```
## Usage
The pipeline can be executed by running the app.py script with the appropriate arguments. The script takes a configuration file and a stage number as arguments.
```bash
python app.py --config <path_to_config_file> --stage <stage_number>
```
## Pipeline Stages
### Stage 1: Normalize Function Call Graphs
This stage normalizes the function call graphs by processing JSON files.
```bash
python app.py --config <path_to_config_file> --stage 1
```
### Stage 2: Train Word2Vec Model
This stage trains a Word2Vec model using the specified parameters.
```bash
python app.py --config <path_to_config_file> --stage 2
```
### Stage 3: Preprocess Graphs
This stage preprocesses the graphs for further analysis and training.
```bash
python app.py --config <path_to_config_file> --stage 3
```
### Stage 4: Train GCN Model
This stage trains and evaluates a Graph Convolutional Network (GCN) with attention.
```bash
python app.py --config <path_to_config_file> --stage 4
```

## Configuration
The pipeline uses a JSON configuration file to manage parameters for each stage. Below is an example structure of the configuration file:
```json
{
  "normalize": {
    "input_dir": "path/to/input",
    "output_dir": "path/to/output"
  },
  "word2vec": {
    "vector_size": 100,
    "epochs": 10,
    "save_path": "path/to/save/word2vec.model",
    "use_gpu": true,
    "graph_dir": "path/to/graphs"
  },
  "graph": {
    "base_directory": "path/to/base",
    "normalized_directory": "path/to/normalized",
    "output_directory": "path/to/output",
    "word2vec_model_path": "path/to/word2vec.model",
    "sample_size": 1000
  },
  "gcn": {
    "num_features": 100,
    "hidden_channels": 64,
    "num_classes": 2,
    "data_dir": "path/to/data",
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "model_save_path": "path/to/save/gcn.model",
    "use_gpu": true
  }
}
```
## License

This project is licensed under the MIT License. See the [LICENSE]() file for more details.

---

By following this README, users can understand the purpose of each stage in the pipeline, how to configure and run the stages, and ensure all necessary dependencies are installed.
