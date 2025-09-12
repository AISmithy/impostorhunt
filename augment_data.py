import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
import os

# Download NLTK data if not already present
nltk.download('wordnet')
nltk.download('omw-1.4')

def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if not synonyms:
            continue
        synonym_words = set()
        for syn in synonyms:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != random_word.lower():
                    synonym_words.add(synonym)
        if len(synonym_words) > 0:
            synonym = random.choice(list(synonym_words))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def augment_csv(input_csv, output_csv, text_col1, text_col2, label_col, augment_factor=1):
    df = pd.read_csv(input_csv)
    augmented_rows = []
    for _, row in df.iterrows():
        for _ in range(augment_factor):
            new_row = row.copy()
            new_row[text_col1] = synonym_replacement(str(row[text_col1]))
            new_row[text_col2] = synonym_replacement(str(row[text_col2]))
            augmented_rows.append(new_row)
    aug_df = pd.DataFrame(augmented_rows)
    result = pd.concat([df, aug_df], ignore_index=True)
    result.to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv}. Original: {len(df)}, Augmented: {len(result)}")

if __name__ == "__main__":
    # Adjust column names as needed
    input_csv = os.path.join('data', 'train.csv')
    output_csv = os.path.join('data', 'train_augmented.csv')
    text_col1 = 'file_1_text'  # Change to your actual column name
    text_col2 = 'file_2_text'  # Change to your actual column name
    label_col = 'label'        # Change to your actual label column name
    augment_factor = 1         # Number of augmented samples per original
    augment_csv(input_csv, output_csv, text_col1, text_col2, label_col, augment_factor)
