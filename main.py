import torch
from transformers import BertTokenizer, BertForSequenceClassification
from model_utils import is_model_saved

from config import Config
from data_loader import create_data_loaders, _read_text_file # Import _read_text_file for final prediction demo
from model_trainer import BertTextPairClassifier

def main():
    # 1. Load Configuration
    config = Config() # Config initializes itself and determines device

    print(f"Using device: {config.DEVICE}")


    # 2. Load Tokenizer and Model (or load from saved if available)
    model_dir = config.OUTPUT_DIR
    if is_model_saved(model_dir):
        print(f"Found saved model in {model_dir}. Loading without retraining...")
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        model.to(config.DEVICE)
        trained = True
    else:
        print("No saved model found. Loading base model and tokenizer for training...")
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
        model.to(config.DEVICE)
        trained = False

    # 3. Prepare DataLoaders
    print("\nPreparing dataset and data loaders...")
    train_dataloader, val_dataloader = create_data_loaders(
        train_csv_path=config.TRAIN_CSV,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        data_dir=config.TRAIN_DIR,
        file1_name=config.FILE_1_NAME,
        file2_name=config.FILE_2_NAME,
        batch_size=config.BATCH_SIZE
    )
    print("Data loaders prepared.")

    # 4. Initialize and Train the Model

    trainer = BertTextPairClassifier(
        model=model,
        tokenizer=tokenizer,
        device=config.DEVICE,
        learning_rate=config.LEARNING_RATE,
        output_dir=config.OUTPUT_DIR,
        max_len=config.MAX_LEN
    )


    if not trained:
        trainer.train(train_dataloader, config.EPOCHS, val_dataloader=val_dataloader)
        # 6. Save Model
        trainer.save_model()
    else:
        print("Skipping training and saving as model is already available.")

    # 5. Evaluate (if validation data exists)
    if val_dataloader is not None:
        print("\nEvaluating model on validation set...")
        trainer.evaluate(val_dataloader)
    else:
        print("\nNo separate validation set available due to insufficient data.")



    # 7. Predict on all test subfolders
    print("\n--- Checking all test articles for REAL vs FAKE ---")
    import os
    test_dir = config.TEST_DIR
    test_results = []
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist.")
    else:
        subfolders = [f.path for f in os.scandir(test_dir) if f.is_dir()]
        if not subfolders:
            print(f"No subfolders found in {test_dir}.")
        for article_dir in sorted(subfolders):
            file1_path = os.path.join(article_dir, config.FILE_1_NAME)
            file2_path = os.path.join(article_dir, config.FILE_2_NAME)
            text1 = _read_text_file(file1_path)
            text2 = _read_text_file(file2_path)
            if text1 and text2:
                prediction_text, confidence = trainer.predict_pair(text1, text2)
                print(f"{os.path.basename(article_dir)}: {prediction_text} (Confidence: {confidence:.4f})")
                test_results.append({
                    'article': os.path.basename(article_dir),
                    'prediction': prediction_text,
                    'confidence': confidence
                })
            else:
                print(f"{os.path.basename(article_dir)}: Could not read both files.")
    summarize_test_results = __import__('model_utils').summarize_test_results
    summary = summarize_test_results(test_results)
    print("\n--- Test Summary ---")
    print(f"Total Articles Processed: {summary['total']}")
    print(f"Articles Predicted as REAL: {summary['real_count']}")
    print(f"Articles Predicted as FAKE: {summary['fake_count']}")
    print(f"Average Confidence: {summary['average_confidence']:.4f}")

    # Save predictions in required CSV format: id,real_text_id
    import pandas as pd
    submission_rows = []
    for result in test_results:
        # Extract article id (e.g., article_0001 -> 1)
        article_str = result['article']
        try:
            article_id = int(article_str.split('_')[-1])
        except Exception:
            article_id = article_str
        # Determine predicted real_text_id from prediction string
        if 'file_1.txt is REAL' in result['prediction']:
            real_text_id = 1
        else:
            real_text_id = 2
        submission_rows.append({'id': article_id, 'real_text_id': real_text_id})
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('submission.csv', index=False)
    print("\nSaved predictions to submission.csv in format: id,real_text_id")
    

if __name__ == "__main__":
    main()