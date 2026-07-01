import os
import sys
import logging

def setup_directories():
    directories = ['data', 'src', 'notebooks', 'outputs', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[INFO] Created directory: {directory}")

def get_logger(name="FitnessML"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(name)