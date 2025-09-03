# main.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from config import Config
from data_loader import create_data_loaders, _read_text_file # Import _read_text_file for final prediction demo
from model_trainer import BertTextPairClassifier

def main():
    # 1. Load Configuration
    config = Config() # Config initializes itself and determines device

    print(f"Using device: {config.DEVICE}")

    # 2. Load Tokenizer and Model
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    print("Tokenizer loaded.")

    print("Loading BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
    model.to(config.DEVICE)
    print("Model loaded.")

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

    trainer.train(train_dataloader, config.EPOCHS)

    # 5. Evaluate (if validation data exists)
    if val_dataloader is not None:
        print("\nEvaluating model on validation set...")
        trainer.evaluate(val_dataloader)
    else:
        print("\nNo separate validation set available due to insufficient data.")


    # 6. Save Model
    trainer.save_model()


    # 7. Example Prediction (using the first available sample from the CSV)
    print("\n--- Example Prediction using the fine-tuned BERT model ---")
    import pandas as pd
    import os
    example_df = pd.read_csv(config.TRAIN_CSV)
    found_example = False
    for _, row in example_df.iterrows():
        sample_id = row['id']
        article_dir = os.path.join(config.TRAIN_DIR, f"article_{str(sample_id).zfill(4)}")
        file1_path = os.path.join(article_dir, config.FILE_1_NAME)
        file2_path = os.path.join(article_dir, config.FILE_2_NAME)
        example_text1_content = _read_text_file(file1_path)
        example_text2_content = _read_text_file(file2_path)
        if example_text1_content and example_text2_content:
            prediction_text, confidence = trainer.predict_pair(example_text1_content, example_text2_content)
            print(f"Predicted outcome: {prediction_text} (Confidence: {confidence:.4f})")
            print(f"True label for this sample: 'file_{row['real_text_id']}.txt' is REAL.")
            found_example = True
            break
    if not found_example:
        print("Could not load any example text files for final prediction demo.")

if __name__ == "__main__":
    main()