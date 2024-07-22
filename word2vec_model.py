import re
import json
import os
import logging
from tqdm import tqdm
from datetime import datetime
from gensim.models import Word2Vec
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] [%(asctime)s] %(message)s',
                    handlers=[logging.FileHandler("word2vec_training.log"),
                              logging.StreamHandler()])


class Word2VecCBOW:
    def __init__(self, vector_size, epochs, save_path, use_gpu):
        """Initialize the Word2Vec model with the specified parameters."""
        self.vector_size = vector_size
        self.epochs = epochs
        self.save_path = save_path
        self.use_gpu = use_gpu
        self.model = None

    def train(self, sentences):
        """Train the model using the provided sentences."""
        if self.use_gpu:
            self.model = Word2Vec(sentences, vector_size=self.vector_size, window=5,
                                  min_count=1, workers=4, sg=0, epochs=self.epochs, compute_loss=True, compute_device="gpu")
        else:
            self.model = Word2Vec(sentences, vector_size=self.vector_size, window=5,
                                  min_count=1, workers=4, sg=0, epochs=self.epochs)
        logging.info("Training completed successfully.")

    def get_word_vectors(self):
        """Return the learned word vectors."""
        return self.model.wv

    def parse_fcg_json_file(self, file_path):
        """Parse a single FCG JSON file and extract tokenized instructions."""
        instructions = []
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for function_key, function_value in data.items():
                function_instructions = function_value.get('instructions', [])
                instructions.extend(function_instructions)
            logging.debug(
                f"Parsed {len(instructions)} instructions from {file_path}.")
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
        tokenized_instructions = [re.findall(
            r'\w+', instruction) for instruction in instructions]
        return tokenized_instructions

    def parallel_process(self, args, batch_size=16):
        """Process the extraction tasks in parallel with batching."""
        results = []

        def process_batch(batch):
            batch_results = []
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(
                    self.parse_fcg_json_file, arg) for arg in batch]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch", unit="file"):
                    try:
                        result = future.result()
                        if result:
                            batch_results.extend(result)
                    except Exception as e:
                        logging.error(f"Error in future result: {e}")
            return batch_results

        # Split the list of args into batches
        for i in range(0, len(args), batch_size):
            batch = args[i:i+batch_size]
            logging.info(
                f"Processing batch {i//batch_size + 1} of {len(args)//batch_size + 1}")
            batch_results = process_batch(batch)
            results.extend(batch_results)

        return results

    def label_and_parse_data(self, base_directory):
        """Label data based on directory structure and parse JSON files."""
        data = []
        for label, subdir in enumerate(['benign', 'malware']):
            directory = os.path.join(base_directory, subdir)
            logging.info(f"Checking directory: {directory}")
            if not os.path.exists(directory):
                logging.warning(f"Directory not found: {directory}")
                continue
            all_files = [os.path.join(root, file) for root, _, files in os.walk(
                directory) for file in files if file.endswith('.json')]
            logging.info(f"Found {len(all_files)} JSON files in {directory}")

            # Process all files
            instructions = self.parallel_process(all_files)
            if not instructions:
                logging.warning(
                    f"No instructions found in directory: {directory}")
                continue
            labeled_instructions = [(instruction, label)
                                    for instruction in instructions]
            data.extend(labeled_instructions)
            # Free memory after processing each subdirectory
            instructions.clear()
        return data

    def save_model(self):
        """Save the model to the specified path."""
        self.model.save(self.save_path)
        logging.info(f"Model saved to {self.save_path}")

    @staticmethod
    def writelog(message):
        """Log messages with timestamp."""
        logging.info(message)

    @staticmethod
    def parse_args():
        """Parse command line arguments."""
        import argparse
        parser = argparse.ArgumentParser(
            description='Assembly Word2Vec Trainer')
        parser.add_argument('-d', '--graph-dir', type=str, required=True, metavar='<directory>',
                            help='base directory containing benign and malware folders')
        parser.add_argument('--modelparams', type=str, required=False, metavar='<path>', default='./model_params/params.json',
                            help='path to model parameters JSON file')
        parser.add_argument('--use-gpu', action='store_true',
                            help='use GPU for training')
        args = parser.parse_args()
        return args

    @classmethod
    def load_model_params(cls, filepath):
        """Load model parameters from a JSON file."""
        with open(filepath, 'r') as file:
            params = json.load(file)
        return params

    def run(self, base_directory):
        """Run the training process."""
        try:
            self.writelog("Starting Word2Vec training")

            labeled_data = self.label_and_parse_data(base_directory)
            if not labeled_data:
                self.writelog("No data found for training. Exiting.")
                return

            # Train Word2Vec model on all instructions
            all_instructions = [
                instruction for instruction, label in labeled_data]
            self.writelog(f'Total instructions: {len(all_instructions)}')

            self.train(all_instructions)
            self.writelog("Training completed")

            self.save_model()
        except Exception as e:
            logging.error(f"Run interrupted or ended with error: {e}")
