# model_trainer.py

import torch
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import os
from data_loader import _read_text_file # Import helper for prediction example

class BertTextPairClassifier:
    """
    Encapsulates the BERT model, tokenizer, training, evaluation, and prediction logic.
    """
    def __init__(self, model, tokenizer, device, learning_rate, output_dir, max_len):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.output_dir = output_dir
        self.max_len = max_len

    def train(self, train_dataloader, epochs):
        """
        Trains the BERT model using the provided DataLoader.
        """
        self.model.train()
        print(f"\nStarting BERT model training on {self.device}...")
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} Training")

            for batch in progress_bar:
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"\nEpoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        print("Training complete.")

    def evaluate(self, dataloader, epoch_num=0):
        """
        Evaluates the model on a given DataLoader.
        """
        if dataloader is None or len(dataloader.dataset) == 0:
            print(f"No data available for evaluation in Epoch {epoch_num}.")
            return None, None

        self.model.eval()
        predictions = []
        true_labels = []
        total_eval_loss = 0

        progress_bar_eval = tqdm(dataloader, desc=f"Evaluation (Epoch {epoch_num})")
        with torch.no_grad():
            for batch in progress_bar_eval:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                total_eval_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_eval_loss = total_eval_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch_num} - Validation Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
        return accuracy, avg_eval_loss

    def predict_pair(self, text1, text2):
        """
        Makes a prediction for a single pair of texts.
        Returns the prediction string and its confidence score.
        """
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).flatten().cpu().numpy()
        
        # Index 1 corresponds to label=1 (file_1 is real), index 0 to label=0 (file_2 is real)
        if probabilities[1] > probabilities[0]:
            return "file_1.txt is REAL, file_2.txt is FAKE", probabilities[1]
        else:
            return "file_2.txt is REAL, file_1.txt is FAKE", probabilities[0]

    def save_model(self):
        """
        Saves the fine-tuned model and tokenizer to the specified output directory.
        """
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Model and tokenizer saved to {self.output_dir}")