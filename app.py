from tools import NormalizeAssembly
from word2vec_model import Word2VecCBOW
from gcn_model import GCNWithAttention
from graph_preprocessing import GraphProcessor
from logging import logging

if __name__ == '__main__':

    """Normalize fcg"""

    args = NormalizeAssembly.parse_arguments()
    normalizer = NormalizeAssembly(args.directory, args.output)
    normalizer.process_json_files()

    """Train Word2Vec"""

    args = Word2VecCBOW.parse_args()
    model_params = Word2VecCBOW.load_model_params(args.modelparams)

    model = Word2VecCBOW(
        vector_size=model_params['vector_size'],
        epochs=model_params['epochs'],
        save_path=model_params['save_path']
    )
    model.run(args.graph_dir)

    """Graph Preprocessing"""

    GraphProcessor.setup_logging()
    processor = GraphProcessor('./model_saved/word2vec')
    labeled_data = processor.label_and_parse_data(
        '/mnt/E/mnt/bigDisk/yishan/dataset_disassemble_normalized/results', sample_size=100)
    G = processor.create_graph(labeled_data)
    data = processor.prepare_data(G)
    logging.info(
        f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    logging.info(
        f"Feature matrix shape: {data.x.shape}, Labels shape: {data.y.shape}")
    logging.info(f"First 5 node features:\n{data.x[:5]}")
    logging.info(f"First 5 node labels:\n{data.y[:5]}")

    """Train Graph Convolution Network"""
