import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import BertForSequenceClassification, BertTokenizerFast
import tensorflow as tf

class MonkeyVerificationPipeline:
    def __init__(self, text_model_dir, image_model_path, class_mapping_path):
        """
        Initialize the pipeline for verifying monkey species in text and image.

        Args:
            text_model_dir: Directory containing the trained text classifier
            image_model_path: Path to the trained image classification model (.h5 file)
            class_mapping_path: Path to the class mapping JSON file
        """
        # Load text classifier
        self.text_model, self.tokenizer, self.id2species = self._load_text_model(text_model_dir)

        # Load image classifier
        self.image_model = self._load_image_model(image_model_path)

        # Load class mapping for image model
        with open(class_mapping_path, 'r') as f:
            self.class_mapping = json.load(f)

        # Set device for PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move text model to device
        self.text_model.to(self.device)

        # Set text model to evaluation mode
        self.text_model.eval()

    def _load_text_model(self, model_dir):
        """Load the text classification model."""
        # Load tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)

        # Load model
        model = BertForSequenceClassification.from_pretrained(model_dir)

        # Load species mappings
        with open(os.path.join(model_dir, 'id2species.json'), 'r') as f:
            id2species = json.load(f)

        # Convert string keys to integers
        id2species = {int(k): v for k, v in id2species.items()}

        return model, tokenizer, id2species

    def _load_image_model(self, model_path):
        """Load the image classification model (.h5 format)."""
        # Load the TensorFlow model
        model = tf.keras.models.load_model(model_path)

        # Print model summary to understand input requirements
        model.summary()

        return model

    def _preprocess_image(self, image_path):
        """Preprocess the image for the model."""
        # Load the image
        img = Image.open(image_path).convert('RGB')

        # Resize to 224x224 (based on error message)
        img = img.resize((224, 224))

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _predict_species_from_text(self, text):
        """Predict monkey species from text."""
        # Tokenize the text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # Get predicted class
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(logits, dim=1).item()

        # Get species name and confidence
        predicted_species = self.id2species[predicted_class]
        confidence = probabilities[predicted_class].item()

        # Get top 3 predictions
        top_k = min(3, len(self.id2species))
        top_probs, top_classes = torch.topk(probabilities, k=top_k)

        top_predictions = []
        for prob, cls in zip(top_probs, top_classes):
            species = self.id2species[cls.item()]
            top_predictions.append((species, prob.item()))

        return predicted_species, confidence, top_predictions

    def _predict_species_from_image(self, image_path):
        """Predict monkey species from image using TensorFlow model."""
        # Preprocess the image
        image_array = self._preprocess_image(image_path)

        # Make prediction
        predictions = self.image_model.predict(image_array)

        # Get predicted class
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # Get species name
        predicted_species = self.class_mapping[str(predicted_class)]

        # Get top 3 predictions
        top_k = min(3, len(self.class_mapping))
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]

        top_predictions = []
        for idx in top_indices:
            species = self.class_mapping[str(idx)]
            prob = float(predictions[0][idx])
            top_predictions.append((species, prob))

        return predicted_species, confidence, top_predictions

    def verify(self, text, image_path):
        """
        Verify if the claimed monkey species in the text matches the species in the image.

        Args:
            text: Text input claiming a monkey species
            image_path: Path to the image

        Returns:
            dict: Dictionary containing verification result and predictions
        """
        # Predict species from text
        text_species, text_confidence, text_top_predictions = self._predict_species_from_text(text)

        # Predict species from image
        image_species, image_confidence, image_top_predictions = self._predict_species_from_image(image_path)

        # Compare predictions
        match = text_species.lower() == image_species.lower()

        # Create result dictionary
        result = {
            'match': match,
            'text_prediction': {
                'species': text_species,
                'confidence': text_confidence,
                'top_predictions': text_top_predictions
            },
            'image_prediction': {
                'species': image_species,
                'confidence': image_confidence,
                'top_predictions': image_top_predictions
            }
        }

        return result

def main():
    parser = argparse.ArgumentParser(description='Monkey Species Verification Pipeline')
    parser.add_argument('--text_model_dir', type=str, default='monkey_classifier', help='Directory with the trained text classifier')
    parser.add_argument('--image_model_path', type=str, default='best_monkey_model.h5', help='Path to the trained image classification model (.h5 file)')
    parser.add_argument('--class_mapping', type=str, default='class_mapping.json', help='Path to the class mapping JSON file')
    parser.add_argument('--text', type=str, required=True, help='Text input claiming a monkey species')
    parser.add_argument('--image', type=str, required=True, help='Path to the image')

    args = parser.parse_args()

    # Initialize the pipeline
    pipeline = MonkeyVerificationPipeline(
        args.text_model_dir,
        args.image_model_path,
        args.class_mapping
    )

    # Verify the claim
    result = pipeline.verify(args.text, args.image)

    # Print results
    print("\nVerification Results:")
    print(f"Text: \"{args.text}\"")
    print(f"Text species prediction: {result['text_prediction']['species']} (confidence: {result['text_prediction']['confidence']:.2%})")
    print("\nTop text predictions:")
    for species, confidence in result['text_prediction']['top_predictions']:
        print(f"  - {species}: {confidence:.2%}")

    print(f"\nImage species prediction: {result['image_prediction']['species']} (confidence: {result['image_prediction']['confidence']:.2%})")
    print("\nTop image predictions:")
    for species, confidence in result['image_prediction']['top_predictions']:
        print(f"  - {species}: {confidence:.2%}")

    print(f"\nMatch: {result['match']}")

    # Return the match result (1 for True, 0 for False)
    return int(result['match'])

if __name__ == "__main__":
    result = main()
    exit(result)