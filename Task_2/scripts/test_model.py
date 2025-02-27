import os
import json
import torch
from transformers import BertForTokenClassification, BertTokenizerFast

def test_model(model_dir, test_text):
    print(f"Loading model from {model_dir}...")

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    print("Tokenizer loaded successfully")

    # Load model
    model = BertForTokenClassification.from_pretrained(model_dir)
    print(f"Model loaded successfully with {model.num_labels} labels")

    # Load tag mappings
    with open(os.path.join(model_dir, 'tag2id.json'), 'r') as f:
        tag2id = json.load(f)

    id2tag = {int(id): tag for tag, id in tag2id.items()}
    print(f"Loaded {len(tag2id)} tags: {list(tag2id.keys())[:5]}...")

    # Tokenize input text
    inputs = tokenizer(
        test_text,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0].numpy()

    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].numpy()

    # Map predictions to tags
    predicted_tags = [id2tag[pred] for pred in predictions]

    # Print results
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print("\nTokens and predicted tags:")

    # Format results for better readability
    for i, (token, tag, (start, end)) in enumerate(zip(tokens, predicted_tags, offset_mapping)):
        if start > 0 and end > 0:  # Skip special tokens
            original_text = test_text[start:end]
            print(f"{i:3d} | {token:15s} | {tag:20s} | '{original_text}'")

    # Extract entities
    entities = []
    current_entity = None

    for i, (tag, (start, end)) in enumerate(zip(predicted_tags, offset_mapping)):
        if start == 0 and end == 0:  # Skip special tokens
            continue

        if tag.startswith('B-'):
            # End previous entity if exists
            if current_entity:
                entities.append(current_entity)

            # Start new entity
            species = tag[2:]  # Remove 'B-' prefix
            current_entity = {
                'text': test_text[start:end],
                'start': int(start),
                'end': int(end),
                'species': species
            }

        elif tag.startswith('I-') and current_entity:
            # Continue current entity
            species = tag[2:]  # Remove 'I-' prefix

            # Only extend if species matches
            if species == current_entity['species']:
                current_entity['text'] = test_text[current_entity['start']:end]
                current_entity['end'] = int(end)

        elif tag == 'O' and current_entity:
            # End current entity
            entities.append(current_entity)
            current_entity = None

    # Add final entity if exists
    if current_entity:
        entities.append(current_entity)

    # Print extracted entities
    print("\nExtracted entities:")
    if entities:
        for entity in entities:
            print(f"- {entity['species']}: '{entity['text']}' at positions {entity['start']}:{entity['end']}")
    else:
        print("No entities detected")

    return entities

if __name__ == "__main__":
    model_dir = "model"

    test_texts = [
        "I observed a group of mantled howler monkeys in the forest.",
        "The pygmy marmoset is one of the smallest primates in the world.",
        "We spotted a japanese macaque bathing in the hot spring.",
        "Can you see the nilgiri langur in that tree?",
        "Look at the patas monkey running across the savanna!",
        "Both the white headed capuchin and common squirrel monkey live in this area."
    ]

    print("Testing model with various examples...\n")

    for i, text in enumerate(test_texts):
        print(f"\n=== Test {i+1}: {text} ===")
        entities = test_model(model_dir, text)