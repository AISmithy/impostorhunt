# config.py

import os
import torch # Imported here to determine device

class Config:
    MODEL_NAME = 'bert-base-uncased' # Pre-trained BERT model name
    MAX_LEN = 512 # Maximum sequence length for BERT tokenizer
    BATCH_SIZE = 8 # Batch size for DataLoader
    EPOCHS = 3 # Number of training epochs
    LEARNING_RATE = 2e-5 # Learning rate for the optimizer
    OUTPUT_DIR = './bert_model_save/' # Directory to save the trained model and tokenizer

    # Paths for dataset structure
    DATA_DIR = 'data'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')  # Path to the training CSV file
    # File names (used for dynamic path construction)
    FILE_1_NAME = 'file_1.txt'
    FILE_2_NAME = 'file_2.txt'

    # Determine the device to use (GPU if available, else CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)