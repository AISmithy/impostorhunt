import unittest
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from model_trainer import BertTextPairClassifier

class TestModelUtils(unittest.TestCase):
    def test_bert_text_pair_classifier_init(self):
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        device = torch.device('cpu')
        learning_rate = 1e-5
        output_dir = '.'
        max_len = 8
        classifier = BertTextPairClassifier(model, tokenizer, device, learning_rate, output_dir, max_len)
        self.assertIsNotNone(classifier)

if __name__ == '__main__':
    unittest.main()
