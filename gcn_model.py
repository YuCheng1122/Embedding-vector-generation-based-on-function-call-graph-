import os
import json
import torch
import gc
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gmp
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    filename='gcn_log.log',
    level=logging.INFO,
    format='[%(levelname)s] [%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class GCNWithAttention(torch.nn.Module):
    """Graph Convolutional Network with Attention."""

    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCNWithAttention, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, 128)
        self.lin2 = Linear(128, 128)
        self.attention = Linear(hidden_channels, 1)
        self.lin = Linear(128, num_classes)

    def forward(self, data_batch):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        logging.info(f"Input x shape: {x.shape}")
        logging.info(f"Edge index shape: {edge_index.shape}")

        x = self.conv1(x, edge_index)
        x = x.relu()
        logging.info(f"Shape after conv1: {x.shape}")

        x = self.conv2(x, edge_index)
        logging.info(f"Shape after conv2: {x.shape}")

        attn_weights = F.leaky_relu(self.attention(x))
        attn_weights = F.softmax(attn_weights, dim=0)
        x = x * attn_weights
        logging.info(f"Shape after attention: {x.shape}")

        x = gmp(x, batch)
        logging.info(f"Shape after global mean pool: {x.shape}")

        x = self.lin1(x)
        x = F.relu(x)
        logging.info(f"Shape after lin1: {x.shape}")

        x = self.lin2(x)
        x = F.relu(x)
        logging.info(f"Shape after lin2: {x.shape}")

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = F.softmax(x, dim=1)
        logging.info(f"Output shape: {x.shape}")

        return x

    @staticmethod
    def load_graph_from_json(json_path):
        """Load a graph from a JSON file and convert it to a PyTorch Geometric Data object."""
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON file {json_path}: {e}")
            raise

        node_features = [node['feature'] for node in data['nodes']]
        x = torch.tensor(node_features, dtype=torch.float)

        edges = [(link['source'], link['target']) for link in data['links']]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        graph_data = Data(x=x, edge_index=edge_index)
        return graph_data

    @staticmethod
    def load_dataset_from_directory(directory, label):
        """Load all JSON files from a directory and assign a label to each graph."""
        dataset = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    try:
                        graph_data = GCNWithAttention.load_graph_from_json(
                            json_path)
                        graph_data.y = torch.tensor([label], dtype=torch.long)
                        dataset.append(graph_data)
                    except json.JSONDecodeError:
                        continue
        return dataset

    def load_dataset(self, data_dir):
        """Load and combine benign and malware datasets."""
        benign_directory = os.path.join(data_dir, 'benign')
        malware_directory = os.path.join(data_dir, 'malware')

        benign_dataset = self.load_dataset_from_directory(
            benign_directory, label=0)
        malware_dataset = self.load_dataset_from_directory(
            malware_directory, label=1)

        dataset = benign_dataset + malware_dataset
        return dataset

    def train_and_evaluate(self, data_dir, params, device):
        """Train and evaluate the model."""
        dataset = self.load_dataset(data_dir)

        # Split the dataset
        train_dataset, test_dataset = train_test_split(
            dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=params['batch_size'], shuffle=False)

        optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'])
        criterion = torch.nn.CrossEntropyLoss()

        trainer = TorchTrainer(
            self, train_loader, test_loader, optimizer, criterion, device)
        trainer.run(params['epochs'])

        torch.save(self.state_dict(), params['model_save_path'])
        print(f'Model saved to {params["model_save_path"]}')


class TorchTrainer:
    """Trainer for PyTorch models."""

    def __init__(self, model, train_loader, test_loader, optimizer, criterion, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
            gc.collect()  # Enforce garbage collection
        return total_loss / len(self.train_loader.dataset)

    def test(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        correct = 0
        for data in self.test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
        return correct / len(self.test_loader.dataset)

    def run(self, epochs):
        """Run the training and evaluation loop for a given number of epochs."""
        for epoch in range(epochs):
            train_loss = self.train()
            test_acc = self.test()
            print(
                f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')


class ConfigLoader:
    """Load configuration parameters from a JSON file."""
    @staticmethod
    def load_model_params(filepath):
        with open(filepath, 'r') as file:
            params = json.load(file)
        return params


def load_graph_from_json(json_path):
    """Load a graph from a JSON file and convert it to a PyTorch Geometric Data object."""
    try:
        with open(json_path, 'r') as file:
            content = file.read().strip()
            if not content:
                logging.warning(f"Empty JSON file: {json_path}")
                return None
            data = json.loads(content)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {json_path}: {e}")
        return None

    node_features = [node['feature'] for node in data['nodes']]
    x = torch.tensor(node_features, dtype=torch.float)
    if x.size(1) != 128:  # Check the number of features
        logging.warning(
            f"Feature size mismatch in file {json_path}. Expected 128, got {x.size(1)}.")

    edges = [(link['source'], link['target']) for link in data['links']]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    graph_data = Data(x=x, edge_index=edge_index)
    return graph_data


def load_dataset_from_directory(directory, label):
    """Load all JSON files from a directory and assign a label to each graph."""
    dataset = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                graph_data = load_graph_from_json(json_path)
                if graph_data is not None:
                    graph_data.y = torch.tensor([label], dtype=torch.long)
                    dataset.append(graph_data)
    return dataset
