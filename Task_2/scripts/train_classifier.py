import os
import json
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report
from tqdm import tqdm

def extract_species_from_data(json_file):
    """Extract unique species from the dataset."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all unique species
    species_set = set()
    for item in data:
        for entity in item['entities']:
            species_set.add(entity['species'])

    # Convert to list and sort for deterministic indexing
    species_list = sorted(list(species_set))

    # Create mapping dictionaries
    species2id = {species: i for i, species in enumerate(species_list)}
    id2species = {i: species for i, species in enumerate(species_list)}

    print(f"Found {len(species_list)} unique monkey species")
    print(f"Species: {species_list}")

    return species_list, species2id, id2species

def prepare_classification_data(json_file, tokenizer, max_length=128):
    """Prepare data for sequence classification."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract unique species
    species_list, species2id, id2species = extract_species_from_data(json_file)

    # Prepare examples for each text containing a monkey species
    examples = []

    for item in data:
        text = item['text']

        # Skip texts with no entities or multiple different species
        species_in_text = set(entity['species'] for entity in item['entities'])

        if len(species_in_text) != 1:
            continue

        species = list(species_in_text)[0]
        label = species2id[species]

        # Tokenize the text
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Add to examples
        examples.append({
            'text': text,
            'input_ids': encoded['input_ids'][0],
            'attention_mask': encoded['attention_mask'][0],
            'label': label,
            'species': species
        })

    print(f"Created {len(examples)} examples for sequence classification")

    # Create label distribution
    label_counts = {}
    for example in examples:
        label = example['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {id2species[label]}: {count} examples")

    return examples, species2id, id2species

def create_data_loaders(examples, test_size=0.2, batch_size=16, seed=42):
    """Split data and create DataLoaders."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Shuffle examples
    indices = np.random.permutation(len(examples))
    split_idx = int(len(examples) * (1 - test_size))

    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create datasets
    train_dataset = TensorDataset(
        torch.stack([examples[i]['input_ids'] for i in train_indices]),
        torch.stack([examples[i]['attention_mask'] for i in train_indices]),
        torch.tensor([examples[i]['label'] for i in train_indices])
    )

    test_dataset = TensorDataset(
        torch.stack([examples[i]['input_ids'] for i in test_indices]),
        torch.stack([examples[i]['attention_mask'] for i in test_indices]),
        torch.tensor([examples[i]['label'] for i in test_indices])
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size
    )

    print(f"Created dataloaders with {len(train_dataloader)} training batches and {len(test_dataloader)} test batches")

    return train_dataloader, test_dataloader, [examples[i] for i in train_indices], [examples[i] for i in test_indices]

def compute_class_weights(examples, num_classes):
    """Compute class weights to handle imbalanced classes."""
    # Count occurrences of each class
    class_counts = {}
    for example in examples:
        label = example['label']
        class_counts[label] = class_counts.get(label, 0) + 1

    # Calculate weights inversely proportional to frequency
    weights = torch.ones(num_classes)
    total_examples = len(examples)

    for label, count in class_counts.items():
        weights[label] = total_examples / (num_classes * count)

    # Normalize weights to avoid extreme values
    weights = weights / weights.sum() * num_classes

    print(f"Class weights: {weights}")
    return weights

def train_classifier(json_file, output_dir, epochs=20, batch_size=16, learning_rate=5e-4, max_length=128):
    """Train a sequence classifier for monkey species identification."""
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Prepare data
    examples, species2id, id2species = prepare_classification_data(json_file, tokenizer, max_length)

    # Create data loaders
    train_dataloader, test_dataloader, train_examples, test_examples = create_data_loaders(
        examples, batch_size=batch_size)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(species2id)
    )

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Compute class weights
    class_weights = compute_class_weights(train_examples, len(species2id))
    class_weights = class_weights.to(device)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Set up scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Set up loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    print(f"Training on {device}...")
    best_accuracy = 0.0
    patience = 5
    no_improvement = 0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            # Unpack batch and move to device
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        true_labels = []
        pred_labels = []

        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Unpack batch and move to device
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            # Add to lists
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

        # Calculate accuracy
        accuracy = np.mean(np.array(true_labels) == np.array(pred_labels))
        print(f"Accuracy: {accuracy:.4f}")

        # Map IDs to species names for reporting
        true_species = [id2species[label] for label in true_labels]
        pred_species = [id2species[label] for label in pred_labels]

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(true_species, pred_species, zero_division=0))

        # Save the model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement = 0
            print(f"New best accuracy: {best_accuracy:.4f} - Saving model...")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save the model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Save species mappings
            with open(os.path.join(output_dir, 'species2id.json'), 'w') as f:
                json.dump(species2id, f)

            with open(os.path.join(output_dir, 'id2species.json'), 'w') as f:
                json.dump(id2species, f)

            print(f"Model and mappings saved to {output_dir}")
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs")

        # Early stopping
        if no_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print(f"Training complete. Best accuracy: {best_accuracy:.4f}")
    return model, tokenizer, species2id, id2species

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train BERT-based sequence classifier for monkey species')
    parser.add_argument('--data_file', type=str, default='data/monkey_ner_dataset.json', help='Path to JSON dataset')
    parser.add_argument('--output_dir', type=str, default='monkey_classifier', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')

    args = parser.parse_args()

    print(f"Current working directory: {os.getcwd()}")
    print(f"Model will be saved to: {os.path.abspath(args.output_dir)}")

    # Train the model
    model, tokenizer, species2id, id2species = train_classifier(
        args.data_file,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )

    print(f"Final model saved to {args.output_dir}")