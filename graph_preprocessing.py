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
from time import time


class GraphProcessor:
    def __init__(self, config):
        """Initialize the graph processor with a configuration dictionary."""
        self.config = config
        self.word2vec_model_path = config['word2vec_model_path']
        self.vector_size = None
        self.load_word2vec_model()

    def load_word2vec_model(self):
        """Load the Word2Vec model."""
        try:
            self.word2vec_model = Word2Vec.load(self.word2vec_model_path)
            self.vector_size = self.word2vec_model.vector_size
            logging.info(
                f"Loaded Word2Vec model with vector size: {self.vector_size}")
        except Exception as e:
            logging.error(f"Error loading Word2Vec model: {e}")
            raise

    def get_node_features(self, instructions):
        """Generate node features using the Word2Vec model."""
        try:
            words = [
                word for instruction in instructions for word in instruction.split()]
            word_embeddings = [self.word2vec_model.wv[word]
                               for word in words if word in self.word2vec_model.wv]
            return np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(self.vector_size)
        except Exception as e:
            logging.error(f"Error getting node features: {e}")
            return np.zeros(self.vector_size)

    def create_graph(self, function_data, edges):
        """Create a graph from function data and edges."""
        try:
            G = nx.DiGraph()
            node_mapping = {}
            node_id = 0
            valid_graph = False

            for func_name, func_details in function_data.items():
                instructions = func_details['instructions']
                node_features = self.get_node_features(instructions)
                if np.any(node_features):  # Check if features are not all zeros
                    G.add_node(
                        node_id, feature=node_features.tolist(), name=func_name)
                    node_mapping[func_name] = node_id
                    node_id += 1
                    valid_graph = True

            for src_func, dest_func in edges:
                if src_func in node_mapping and dest_func in node_mapping:
                    src_node_id = node_mapping[src_func]
                    dest_node_id = node_mapping[dest_func]
                    G.add_edge(src_node_id, dest_node_id)

            return G if valid_graph else None
        except Exception as e:
            logging.error(f"Error creating graph: {e}")
            return None

    def save_graph(self, G, output_path):
        """Save the graph to a JSON file."""
        try:
            graph_data = nx.readwrite.json_graph.node_link_data(G)
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving graph to JSON: {e}")

    def process_files(self, sample_size=None):
        """Process files to create graphs from function call data."""
        base_directory = self.config['base_directory']
        normalized_directory = self.config['normalized_directory']
        output_directory = self.config['output_directory']
        sample_size = sample_size or self.config.get('sample_size', 1000)

        data_files = []
        saved_files = []
        skipped_files = []

        for subdir in ['benign', 'malware']:
            directory = os.path.join(base_directory, subdir)
            normalized_subdir = os.path.join(normalized_directory, subdir)
            if not os.path.exists(directory):
                continue
            all_files = [os.path.join(root, file) for root, _, files in os.walk(
                directory) for file in files if file.endswith('.dot')]
            valid_files = [file for file in all_files if os.path.exists(
                os.path.join(normalized_subdir, os.path.relpath(file, start=directory)).replace('.dot', '.json'))]

            data_files.extend(random.sample(
                valid_files, min(sample_size, len(valid_files))))

        os.makedirs(output_directory, exist_ok=True)

        start_time = time()
        for file_path in tqdm(data_files, desc="Processing files"):
            try:
                normalized_file_path = os.path.join(
                    normalized_directory, os.path.relpath(file_path, start=base_directory)).replace('.dot', '.json')
                with open(normalized_file_path, 'r') as f:
                    function_data = json.load(f)

                with open(file_path, 'r') as f:
                    dot_content = f.read()

                nodes, edges = self.parse_dot_file(dot_content)
                G = self.create_graph(function_data, edges)
                if G is not None:
                    output_path = os.path.join(output_directory, os.path.relpath(
                        file_path, start=base_directory)).replace('.dot', '.json')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    self.save_graph(G, output_path)
                    saved_files.append(file_path)
                else:
                    skipped_files.append(file_path)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                skipped_files.append(file_path)

        end_time = time()
        logging.info(f"Total files processed: {len(data_files)}")
        logging.info(f"Files saved: {len(saved_files)}")
        logging.info(f"Files skipped: {len(skipped_files)}")
        logging.info(f"Skipped files: {skipped_files}")
        logging.info(f"Processing took {end_time - start_time:.2f} seconds")

    def parse_dot_file(self, content):
        """Parse a .dot file to extract function calls."""
        nodes = {}
        edges = []
        try:
            lines = content.splitlines()

            for line in lines:
                node_match = re.search(
                    r'\"([^\"]+)\"\s+\[label=\"([^\"]+)\"\];', line)
                if node_match:
                    node_id = node_match.group(1)
                    node_label = node_match.group(2)
                    nodes[node_id] = node_label

            for line in lines:
                edge_match = re.search(
                    r'\"([^\"]+)\"\s+->\s+\"([^\"]+)\";', line)
                if edge_match:
                    src_node = edge_match.group(1)
                    dest_node = edge_match.group(2)
                    if src_node in nodes and dest_node in nodes:
                        edges.append((nodes[src_node], nodes[dest_node]))
        except Exception as e:
            logging.error(f"Error parsing .dot file: {e}")
        return nodes, edges

    @staticmethod
    def setup_logging(log_file="graph_processor.log"):
        """Set up logging for the script."""
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
