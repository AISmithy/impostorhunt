## Data Preparation Pipeline

To get the best results, use the following data pipeline:

1. **Extract Text from Files:**
	- Run `build_text_csv.py` to create `data/train_text.csv` with the actual text and labels.
	- Example:
	  ```sh
	  python build_text_csv.py
	  ```

2. **Data Augmentation:**
	- Run `augment_data.py` to generate `data/train_augmented.csv` with additional training samples.
	- Example:
	  ```sh
	  python augment_data.py
	  ```

3. **Configure Training:**
	- In `config.py`, set `TRAIN_CSV` to `'data/train_augmented.csv'` for training with augmented data.

4. **Train and Predict:**
	- Run `main.py` to train the model and make predictions on the test set.
	- Example:
	  ```sh
	  python main.py
	  ```

This pipeline ensures your model is trained on the most diverse and complete data available.
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
2. (Optional but recommended) Run `build_text_csv.py` to extract text from files and create `data/train_text.csv`:
	```sh
	python build_text_csv.py
	```
3. (Optional but recommended) Run `augment_data.py` to generate augmented training data in `data/train_augmented.csv`:
	```sh
	python augment_data.py
	```
4. Ensure `config.py` points to the correct training CSV (e.g., `train_augmented.csv` for best results).
5. Run `main.py` to train (if needed) and predict on all test samples:
	```sh
	python main.py
	```
6. The script will print predictions and a summary of results.


## Requirements
- Python 3.8+
- PyTorch
- Transformers
- tqdm
- pandas
- nltk

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Suppressing Transformers Warnings
To suppress verbose warnings from the Hugging Face Transformers library (e.g., about overflowing tokens), add this to the top of your main script:
```python
from transformers import logging
logging.set_verbosity_error()
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
