import os
import json
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report
from tqdm import tqdm

def load_processed_data(data_dir):
    """Load the processed data for training."""
    print(f"Loading data from: {os.path.abspath(data_dir)}")

    # Load tag mappings
    with open(os.path.join(data_dir, 'tag2id.json'), 'r') as f:
        tag2id = json.load(f)

    id2tag = {int(id): tag for tag, id in tag2id.items()}
    print(f"Loaded {len(tag2id)} unique tags")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(data_dir, 'tokenizer'))
    print(f"Loaded tokenizer from {os.path.join(data_dir, 'tokenizer')}")

    # Load the processed train and test data with weights_only=False
    train_data = torch.load(os.path.join(data_dir, 'train_data.pt'), weights_only=False)
    test_data = torch.load(os.path.join(data_dir, 'test_data.pt'), weights_only=False)
    print(f"Loaded training data with {len(train_data['tags'])} examples")
    print(f"Loaded test data with {len(test_data['tags'])} examples")

    return {
        'train': train_data,
        'test': test_data,
        'tag2id': tag2id,
        'id2tag': id2tag,
        'tokenizer': tokenizer
    }

def create_dataloaders(processed_data, batch_size=16):
    """Create DataLoaders for training and testing."""
    train_dataset = TensorDataset(
        processed_data['train']['inputs']['input_ids'],
        processed_data['train']['inputs']['attention_mask'],
        torch.tensor(processed_data['train']['tags'])
    )

    test_dataset = TensorDataset(
        processed_data['test']['inputs']['input_ids'],
        processed_data['test']['inputs']['attention_mask'],
        torch.tensor(processed_data['test']['tags'])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    print(f"Created dataloaders - Training: {len(train_dataloader)} batches, Testing: {len(test_dataloader)} batches")
    return train_dataloader, test_dataloader

def train_model(processed_data, output_dir, epochs=10, batch_size=16, learning_rate=5e-5):
    """Train the BERT model for NER."""
    # Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(processed_data, batch_size)

    # Initialize model
    model = BertForTokenClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(processed_data['tag2id'])
    )
    print(f"Initialized model with {len(processed_data['tag2id'])} output labels")

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    print(f"Training on {device}...")
    best_f1 = 0.0

    # Print a few examples of the training data to check format
    print("\nChecking training data format:")
    for i, (input_ids, attention_mask, tags) in enumerate(train_dataloader):
        if i == 0:  # Just check the first batch
            print(f"Input shape: {input_ids.shape}")
            print(f"Tags shape: {tags.shape}")
            # Print a few tokens and their tags
            for j in range(min(3, len(input_ids))):  # Look at up to 3 examples
                sample_tokens = processed_data['tokenizer'].convert_ids_to_tokens(input_ids[j][:20])
                sample_tags = [processed_data['id2tag'][tag.item()] for tag in tags[j][:20]]
                print(f"\nExample {j+1} (first 20 tokens):")
                for token, tag in zip(sample_tokens, sample_tags):
                    print(f"{token}: {tag}")
            break

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Unpack the batch and move to device
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        true_labels = []
        predictions = []

        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Unpack the batch and move to device
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Get predictions
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=2)

            # Convert IDs to labels
            for i, label in enumerate(labels):
                true_label = label.cpu().numpy()
                pred = pred_ids[i].cpu().numpy()
                mask = attention_mask[i].cpu().numpy()

                for j in range(len(true_label)):
                    if mask[j]:  # Skip padded tokens
                        true_labels.append(processed_data['id2tag'][true_label[j]])
                        predictions.append(processed_data['id2tag'][pred[j]])

        # Debug: Check prediction distribution
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        print(f"\nPrediction distribution: {pred_counts}")

        true_counts = {}
        for label in true_labels:
            true_counts[label] = true_counts.get(label, 0) + 1
        print(f"True label distribution: {true_counts}")

        # Check if all predictions are "O"
        print(f"Percent 'O' predictions: {pred_counts.get('O', 0) / len(predictions) * 100:.2f}%")

        # Look at a few examples to see what's wrong
        print("\nSample predictions vs true labels:")
        for i in range(min(20, len(true_labels))):
            print(f"True: {true_labels[i]}, Pred: {predictions[i]}")

        # Calculate metrics
        report = classification_report(true_labels, predictions, output_dict=True)

        # Filter out the 'O' tag when calculating F1
        f1_scores = []
        for tag, metrics in report.items():
            if isinstance(metrics, dict) and tag != 'O' and tag.startswith('B-'):
                f1_scores.append(metrics['f1-score'])
                print(f"F1 score for {tag}: {metrics['f1-score']:.4f}")

        avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
        print(f"Epoch {epoch+1} - Average F1 score (B-tags): {avg_f1:.4f}")

        # Save the model
        print(f"Attempting to save model to: {os.path.abspath(output_dir)}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Directory exists after makedirs: {os.path.exists(output_dir)}")

        # Always save the latest model
        model.save_pretrained(output_dir)
        processed_data['tokenizer'].save_pretrained(output_dir)

        # Save the tag mappings
        with open(os.path.join(output_dir, 'tag2id.json'), 'w') as f:
            json.dump(processed_data['tag2id'], f)

        print(f"Model saved. Directory contents: {os.listdir(output_dir)}")

        # Track best model
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            print(f"New best F1 score: {best_f1:.4f}")

    print(f"Training complete. Best F1 score: {best_f1:.4f}")
    return model, best_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT-based NER model')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='model', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for training')

    args = parser.parse_args()

    print(f"Current working directory: {os.getcwd()}")
    print(f"Model will be saved to: {os.path.abspath(args.output_dir)}")

    # Load processed data
    processed_data = load_processed_data(args.data_dir)

    # Train the model
    model, best_f1 = train_model(
        processed_data,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    print(f"Model saved to {args.output_dir}")
    print(f"Final directory contents: {os.listdir(args.output_dir) if os.path.exists(args.output_dir) else 'Directory not found!'}")