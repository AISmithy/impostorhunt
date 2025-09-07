# ImpostorHunt

ImpostorHunt is a machine learning project designed to detect fake texts in a dataset using a fine-tuned BERT model. For each data sample, the system receives two texts: one real and one fake. The model is trained to distinguish between them and predict which text is real.

## Features
- Fine-tunes a BERT model for text pair classification.
- Automatically skips training if a saved model is found.
- Loads and predicts on all test samples in the `data/test` directory.
- Provides a summary of predictions, including counts and average confidence.

## Project Structure
- `main.py`: Main script for training, evaluation, and prediction.
- `model_trainer.py`: Contains the BERT training and prediction logic.
- `data_loader.py`: Handles data loading and preprocessing.
- `config.py`: Configuration for paths, model, and training parameters.
- `model_utils.py`: Utility functions for model management and result summarization.
- `data/`: Contains training and test data.
- `bert_model_save/`: Directory where the trained model and tokenizer are saved.

## Usage
1. Place your training data in `data/train` and test data in `data/test` (each article in its own subfolder with `file_1.txt` and `file_2.txt`).
2. Run `main.py` to train (if needed) and predict on all test samples.
3. The script will print predictions and a summary of results.

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- tqdm
- pandas

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Example Output
```
article_0001: file_1.txt is REAL, file_2.txt is FAKE (Confidence: 0.9876)
article_0002: file_2.txt is REAL, file_1.txt is FAKE (Confidence: 0.9123)
...
Summary: 100 total, 55 real, 45 fake, avg confidence: 0.95
```

---

For more details, see the code and comments in each file.
