import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

def load_data(file_path):
    """Load the NER dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def convert_to_spacy_format(data):
    """Convert the dataset into a format suitable for training."""
    training_data = []

    for item in data:
        text = item['text']
        entities = []

        for entity in item['entities']:
            entities.append((entity['start'], entity['end'], entity['species']))

        training_data.append((text, {"entities": entities}))

    return training_data

def convert_to_bert_format(data, tokenizer, max_length=128):
    """
    Convert data to the format needed for BERT-based NER.
    Using BIO tagging scheme (B-prefix for beginning of entity, I-prefix for inside, O for outside)
    """
    features = []

    for item in data:
        text = item['text']
        entities = item['entities']

        # Tokenize the text and get offsets
        tokenized = tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True  # Explicitly request special tokens mask
        )

        # Extract the offsets
        offset_mapping = tokenized.pop('offset_mapping')[0].numpy()
        special_tokens_mask = tokenized.pop('special_tokens_mask')[0].numpy()

        # Initialize tags as 'O' for all tokens
        tags = ['O'] * len(offset_mapping)

        # Create entity labels
        for entity in entities:
            start_char, end_char = entity['start'], entity['end']
            species = entity['species']

            token_start, token_end = None, None

            # Find start token
            for i, (start_offset, end_offset) in enumerate(offset_mapping):
                # Skip special tokens
                if special_tokens_mask[i] == 1:
                    continue

                # Token starts before or at entity start and ends after entity start
                if start_offset <= start_char and end_offset > start_char:
                    token_start = i
                    break

            # Find end token
            for i, (start_offset, end_offset) in enumerate(offset_mapping):
                # Skip special tokens
                if special_tokens_mask[i] == 1:
                    continue

                # Token starts before entity end and ends at or after entity end
                if start_offset < end_char and end_offset >= end_char:
                    token_end = i
                    break

            # Only assign tags if we found valid token positions
            if token_start is not None and token_end is not None:
                # Mark the first token as B-species
                tags[token_start] = f'B-{species}'

                # Mark the rest of the tokens as I-species
                for i in range(token_start + 1, token_end + 1):
                    if special_tokens_mask[i] == 0:  # Skip special tokens
                        tags[i] = f'I-{species}'

        # Convert tags to IDs
        inputs = {
            'input_ids': tokenized['input_ids'][0].numpy(),
            'attention_mask': tokenized['attention_mask'][0].numpy(),
        }

        features.append({
            'inputs': inputs,
            'tags': tags
        })

    # Combine all features
    combined = {
        'inputs': {
            'input_ids': torch.tensor([f['inputs']['input_ids'] for f in features]),
            'attention_mask': torch.tensor([f['inputs']['attention_mask'] for f in features]),
        },
        'tags': [f['tags'] for f in features]
    }

    return combined

def prepare_data_for_training(file_path, test_size=0.2, max_length=128):
    """Prepare the data for training and testing."""
    data = load_data(file_path)

    # Split into train and test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Create label list
    unique_tags = set()
    for item in data:
        for entity in item['entities']:
            species = entity['species']
            unique_tags.add(f'B-{species}')
            unique_tags.add(f'I-{species}')
    unique_tags = sorted(list(unique_tags))
    unique_tags = ['O'] + unique_tags

    # Create a mapping from tags to IDs
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    # Convert data to BERT format
    train_processed = convert_to_bert_format(train_data, tokenizer, max_length)
    test_processed = convert_to_bert_format(test_data, tokenizer, max_length)

    # Convert tags to IDs
    train_tag_ids = [[tag2id.get(tag, tag2id['O']) for tag in tags] for tags in train_processed['tags']]
    test_tag_ids = [[tag2id.get(tag, tag2id['O']) for tag in tags] for tags in test_processed['tags']]

    train_processed['tags'] = train_tag_ids
    test_processed['tags'] = test_tag_ids

    return {
        'train': train_processed,
        'test': test_processed,
        'tag2id': tag2id,
        'id2tag': id2tag,
        'tokenizer': tokenizer
    }

# Save the processed data
def save_processed_data(processed_data, output_dir):
    """Save the processed data and metadata for training."""
    os.makedirs(output_dir, exist_ok=True)

    # Save tag mappings
    with open(os.path.join(output_dir, 'tag2id.json'), 'w') as f:
        json.dump(processed_data['tag2id'], f)

    # Save tokenizer
    processed_data['tokenizer'].save_pretrained(os.path.join(output_dir, 'tokenizer'))

    # Save the processed train and test data
    torch.save(processed_data['train'], os.path.join(output_dir, 'train_data.pt'))
    torch.save(processed_data['test'], os.path.join(output_dir, 'test_data.pt'))

if __name__ == "__main__":
    import torch

    # Set up paths
    data_dir = os.path.join('data')
    output_dir = os.path.join('processed_data')
    os.makedirs(output_dir, exist_ok=True)

    # Process data
    processed_data = prepare_data_for_training(
        os.path.join(data_dir, 'monkey_ner_dataset.json'),
        test_size=0.2,
        max_length=128
    )

    # Save processed data
    save_processed_data(processed_data, output_dir)

    print(f"Processed data saved to {output_dir}")
    print(f"Number of unique tags: {len(processed_data['tag2id'])}")
    print(f"Tags: {list(processed_data['tag2id'].keys())}")