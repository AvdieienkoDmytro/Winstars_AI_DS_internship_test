import os
import json
import torch
import argparse
from transformers import BertForSequenceClassification, BertTokenizerFast

def test_monkey_classifier(model_dir, test_texts):
    """Test the monkey species classifier on sample texts."""
    print(f"Loading model from {model_dir}...")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    print("Tokenizer loaded successfully")

    # Load model
    model = BertForSequenceClassification.from_pretrained(model_dir)
    print(f"Model loaded successfully with {model.num_labels} labels")

    # Load species mappings
    with open(os.path.join(model_dir, 'id2species.json'), 'r') as f:
        id2species = json.load(f)

    # Convert string keys to integers
    id2species = {int(k): v for k, v in id2species.items()}
    print(f"Loaded {len(id2species)} species")

    # Set model to evaluation mode
    model.eval()

    # Process each test text
    for i, text in enumerate(test_texts):
        print(f"\n=== Test {i+1}: {text}")

        # Tokenize the text
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted class
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(logits, dim=1).item()

        # Get species name
        predicted_species = id2species[predicted_class]
        probability = probabilities[predicted_class].item()

        print(f"Predicted species: {predicted_species} (confidence: {probability:.2%})")

        # Print top 3 species
        top_probs, top_classes = torch.topk(probabilities, k=min(3, len(id2species)))
        print("Top 3 predictions:")
        for prob, cls in zip(top_probs, top_classes):
            species = id2species[cls.item()]
            print(f"  - {species}: {prob.item():.2%}")

if __name__ == "__main__":
    model_dir = "monkey_classifier"

    test_texts = [
        "I observed a group of monkeys in the forest.",
        "The small primate is one of the smallest primates in the world.",
        "We spotted a monkey bathing in the hot spring.",
        "Can you see the monkey in that tree?",
        "Look at the monkey running across the savanna!",
        "Both the white headed and common monkeys live in this area."
    ]

    print("Testing monkey classifier...\n")
    test_monkey_classifier(model_dir, test_texts)