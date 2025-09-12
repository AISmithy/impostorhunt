import unittest
from data_loader import create_data_loaders
from transformers import BertTokenizer
import os

class TestDataLoader(unittest.TestCase):
    def test_create_data_loaders(self):
        # Use a small batch size and max_len for test speed
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_csv_path = os.path.join('data', 'train.csv')
        data_dir = 'data'
        file1_name = 'file_1.txt'
        file2_name = 'file_2.txt'
        batch_size = 2
        max_len = 8
        train_loader, val_loader = create_data_loaders(
            train_csv_path, tokenizer, max_len, data_dir, file1_name, file2_name, batch_size, test_split=0.1
        )
        self.assertIsNotNone(train_loader)

if __name__ == '__main__':
    unittest.main()
