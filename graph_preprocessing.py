import os
import json
import re
import random
import logging
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from torch_geometric.utils import from_networkx
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class GraphProcessor:
    def __init__(self, word2vec_model_path):
        # Load the Word2Vec model
        self.word2vec_model = Word2Vec.load(word2vec_model_path)
        self.vector_size = self.word2vec_model.vector_size

    def get_node_features(self, instructions):
        # Flatten instructions and get embeddings
        words = [word for instruction in instructions for word in instruction]
        word_embeddings = [self.word2vec_model.wv[word]
                           for word in words if word in self.word2vec_model.wv]
        return np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(self.vector_size)

    def create_graph(self, labeled_data):
        # Create a graph with nodes having feature vectors and labels
        G = nx.Graph()
        for i, (instructions, label) in enumerate(labeled_data):
            G.add_node(i, feature=self.get_node_features(
                instructions), label=label)
        G.add_edges_from((i, j) for i in range(len(labeled_data))
                         for j in range(i + 1, len(labeled_data)))
        return G

    def prepare_data(self, G):
        # Prepare data for GCN training
        data = from_networkx(G)
        data.x = torch.tensor([G.nodes[node]['feature']
                              for node in G.nodes], dtype=torch.float)
        data.y = torch.tensor([G.nodes[node]['label']
                              for node in G.nodes], dtype=torch.long)
        nodes = list(G.nodes())
        train_nodes, test_nodes = train_test_split(
            nodes, test_size=0.2, random_state=42)
        train_nodes, val_nodes = train_test_split(
            train_nodes, test_size=0.2, random_state=42)
        data.train_mask, data.val_mask, data.test_mask = [
            torch.tensor([i in mask for i in nodes], dtype=torch.bool) for mask in [train_nodes, val_nodes, test_nodes]
        ]
        return data

    def label_and_parse_data(self, base_directory, sample_size=100):
        # Label and parse data from JSON files
        data = []
        for label, subdir in enumerate(['benign', 'malware']):
            directory = os.path.join(base_directory, subdir)
            if not os.path.exists(directory):
                logging.warning(f"Directory not found: {directory}")
                continue
            all_files = [os.path.join(root, file) for root, _, files in os.walk(
                directory) for file in files if file.endswith('.json')]
            selected_files = random.sample(
                all_files, min(sample_size, len(all_files)))
            for file_path in tqdm(selected_files, desc=f"Processing {subdir} files"):
                with open(file_path, 'r') as f:
                    instructions = [re.findall(
                        r'\w+', ins) for func in json.load(f).values() for ins in func.get('instructions', [])]
                data.append((instructions, label))
        return data

    @staticmethod
    def setup_logging(log_file="graph_processor.log"):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
