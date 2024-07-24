import argparse
import json
import logging
import torch
from tqdm import tqdm
from normalize import NormalizeAssembly
from word2vec_model import Word2VecCBOW
from gcn_model import GCNWithAttention
from graph_preprocessing import GraphProcessor


def load_params(file_path):
    """Load parameters from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def normalize_fcg(params):
    """Normalize function call graphs."""
    normalizer = NormalizeAssembly(params['input_dir'], params['output_dir'])
    normalizer.process_json_files()


def train_word2vec(params):
    """Train Word2Vec model."""
    try:
        model = Word2VecCBOW(
            vector_size=params['vector_size'],
            epochs=params['epochs'],
            save_path=params['save_path'],
            use_gpu=params['use_gpu']
        )
        model.run(params['graph_dir'])
    except Exception as e:
        logging.error(
            f"Word2Vec training interrupted or ended with error: {e}")


def preprocess_graphs(params):
    """Graph preprocessing."""
    GraphProcessor.setup_logging()

    try:
        processor = GraphProcessor(params)
        processor.process_files()
    except Exception as e:
        logging.error(f"Error in graph preprocessing: {e}")


def train_gcn_model(params, device):
    """Train and evaluate the GCN model."""
    model = GCNWithAttention(
        params['num_features'], params['hidden_channels'], params['num_classes']).to(device)

    model.train_and_evaluate(params['data_dir'], params, device)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run stages of the pipeline.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the parameters JSON file.')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3, 4], required=True,
                        help='Stage to run: 1 for normalize, 2 for word2vec, 3 for preprocess graphs, 4 for GCN model.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load parameters
    all_params = load_params(args.config)

    normalize_params = all_params['normalize']
    word2vec_params = all_params['word2vec']
    graph_params = all_params['graph']
    gcn_params = all_params['gcn']

    # Determine device
    device = torch.device(
        'cuda' if gcn_params['use_gpu'] and torch.cuda.is_available() else 'cpu')

    # Execute selected stage
    if args.stage == 1:
        normalize_fcg(normalize_params)
    elif args.stage == 2:
        train_word2vec(word2vec_params)
    elif args.stage == 3:
        preprocess_graphs(graph_params)
    elif args.stage == 4:
        train_gcn_model(gcn_params, device)


if __name__ == '__main__':
    main()
