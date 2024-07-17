import os
import re
import logging
import json
import argparse
from tqdm import tqdm
import networkx as nx
import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed


class NormalizeAssembly:
    def __init__(self, directory, output_dir) -> None:
        self.directory = directory
        self.output_dir = output_dir
        self.extraction_logger, self.timing_logger = self.configure_logging(
            output_dir)

    @staticmethod
    def configure_logging(output_dir: str) -> tuple:
        """
        Configure logging settings.
        """
        os.makedirs(
            output_dir, exist_ok=True)  # Ensure the output directory exists
        extraction_log_file = os.path.join(output_dir, 'extraction.log')
        timing_log_file = os.path.join(output_dir, 'timing.log')

        extraction_logger = logging.getLogger('extraction_logger')
        extraction_logger.setLevel(logging.INFO)
        extraction_handler = logging.FileHandler(extraction_log_file)
        extraction_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        extraction_logger.addHandler(extraction_handler)

        timing_logger = logging.getLogger('timing_logger')
        timing_logger.setLevel(logging.INFO)
        timing_handler = logging.FileHandler(timing_log_file)
        timing_handler.setFormatter(
            logging.Formatter('%(asctime)s,%(message)s'))
        timing_logger.addHandler(timing_handler)

        return extraction_logger, timing_logger

    def normalize(self, inst):
        """
        Normalize a given assembly instruction. According to the paper should have:
        1. Remove all comments;
        2. Replace all numeric constant values with "N";
        3. Replace all irregular strings with "M";
        4. Replace all function names with their short name;
        For example, replace "sub_406492" with "sub", replace "loc_100080CF" with "loc".
        5. Connect the opcode and operand with "-"
        """
        inst = re.sub(r';.*', '', inst)  # Remove comments
        # Replace numeric constant values with "N"
        inst = re.sub(r'\b0x[0-9a-fA-F]+\b|\b\d+\b', 'N', inst)
        # Replace function names with their short name
        inst = re.sub(
            r'\b(sym|fcn|sub|loc|str|reloc|obj)_[0-9a-fA-F]+\b', r'\1', inst)
        # Replace remaining function addresses with "fcn"
        inst = re.sub(r'\bfcn\.[0-9a-fA-F]+\b', 'fcn', inst)

        inst = inst.replace(',', ' ')

        parts = re.split(r'(\s+|[\[\]+*])', inst)
        asm_normed = []
        for part in parts:
            if re.match(r'\b[^a-zA-Z0-9]+\b', part):
                part = 'M'
            elif part.strip():
                asm_normed.append(part.strip())

        normalized_inst = '-'.join(filter(None, asm_normed))

        normalized_inst = re.sub(r'-(\s*-)+', '-', normalized_inst)
        normalized_inst = re.sub(r'\[-|-\]', '', normalized_inst)
        normalized_inst = re.sub(r'-(,)-', '-', normalized_inst)
        normalized_inst = re.sub(r'-+', '-', normalized_inst)
        normalized_inst = re.sub(r'-$', '', normalized_inst)
        normalized_inst = re.sub(r'^-', '', normalized_inst)

        return normalized_inst

    def process_json_files(self):
        """
        Process .json files in the directory, normalize them, and remove empty or unwanted files.
        """
        all_files = []
        for root, _, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.json'):
                    all_files.append(os.path.join(root, file))

        for file_path in tqdm(all_files, desc="Processing JSON files"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            if not data or self.contains_unwanted_pattern(data):
                self.extraction_logger.info(
                    f"Removing unwanted file: {file_path}")
                os.remove(file_path)
            else:
                self.extraction_logger.info(f"Processing file: {file_path}")
                normalized_data = self.normalize_data(data)
                output_file_path = os.path.join(
                    self.output_dir, os.path.relpath(file_path, self.directory))
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, 'w') as f:
                    json.dump(normalized_data, f)

    @staticmethod
    def contains_unwanted_pattern(data):
        """
        Check if the JSON data contains the unwanted pattern.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                if "function_address" in value and "instructions" in value:
                    if value["function_address"].startswith("0x") and value["instructions"] == ["error"]:
                        return True
        return False

    def normalize_data(self, data):
        """
        Normalize data contained in the JSON file.
        """
        normalized_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                normalized_data[key] = self.normalize(value)
            elif isinstance(value, list):
                normalized_data[key] = [self.normalize(
                    v) if isinstance(v, str) else v for v in value]
            elif isinstance(value, dict):
                normalized_data[key] = self.normalize_data(value)
            else:
                normalized_data[key] = value
        return normalized_data

    @staticmethod
    def apply_norm(graph):
        normed_asm = []
        for assembly in nx.get_node_attributes(graph, 'x').values():
            normed_asm.append([NormalizeAssembly().normalize(asm)
                              for addr, asm in assembly])
        return normed_asm

    @staticmethod
    def concat_normed(normed_asm_list):
        return sum(normed_asm_list, [])

    @staticmethod
    def parse_arguments():
        """
        Parse command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description='Normalize assembly instructions in JSON files.')
        parser.add_argument('-d', '--directory', type=str,
                            required=True, help='Path to the input directory')
        parser.add_argument('-o', '--output', type=str,
                            help='Path to the output directory')
        args = parser.parse_args()
        args.directory = os.path.normpath(os.path.expanduser(args.directory))
        if args.output is None:
            input_dir_name = os.path.basename(os.path.normpath(args.directory))
            args.output = os.path.join(os.path.dirname(
                args.directory), f"{input_dir_name}_normalized")
        args.output = os.path.normpath(os.path.expanduser(args.output))
        return args
