import json
import os

# Define the mapping based on your model's training classes
# Make sure these are in the exact same order as during training
monkey_species = [
    "mantled howler",
    "patas monkey",
    "bald uakari",
    "japanese macaque",
    "pygmy marmoset",
    "white headed capuchin",
    "silvery marmoset",
    "common squirrel monkey",
    "black headed night monkey",
    "nilgiri langur"
]

# Create a dictionary mapping class indices to species names
class_mapping = {str(i): species for i, species in enumerate(monkey_species)}

# Save to the project root directory
with open('class_mapping.json', 'w') as f:
    json.dump(class_mapping, f, indent=2)

print(f"Created class_mapping.json with {len(monkey_species)} species")
print(f"File saved to: {os.path.abspath('class_mapping.json')}")