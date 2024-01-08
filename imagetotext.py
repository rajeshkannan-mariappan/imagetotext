import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained ResNet model for image classification
model = models.resnet50(pretrained=True)
model.eval()

# Define a transformation for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
image_path = "/sample home.jfif"
image = Image.open(image_path)
input_tensor = transform(image).unsqueeze(0)

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)
import requests

# Download the labels from the GitHub repository
labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(labels_url)
labels = response.json()

# Save the labels to a local file
with open("imagenet_labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

# Load the labels used by the pre-trained ResNet model
with open("imagenet_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Get the predicted class index
_, predicted_class_idx = torch.max(output, 1)

# Display the result
predicted_label = labels[predicted_class_idx.item()]
print(f"Predicted label: {predicted_label}")
