import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random

# --- STEP 1: DEFINE THE SAME MODEL STRUCTURE ---
# (We must copy this because PyTorch needs to know the shape of the brain)
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 2) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc1(x)
        return x

# --- STEP 2: LOAD THE SAVED BRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)

# Load the weights from the file you just created
model.load_state_dict(torch.load("brain_tumor_model.pth"))
model.eval() # Tell the model we are Testing, not Training
print("Model loaded successfully!")

# --- STEP 3: PICK A RANDOM IMAGE TO TEST ---
# We will pick a random image from your Data folder
data_path = 'Data' 
categories = ['yes', 'no']

# Pick a random category (Tumor or Healthy)
true_label = random.choice(categories)
folder_path = os.path.join(data_path, true_label)

# Pick a random file inside that folder
filename = random.choice(os.listdir(folder_path))
image_path = os.path.join(folder_path, filename)

print(f"\nTesting on image: {filename}")
print(f"Actual Reality: {true_label.upper()}")

# --- STEP 4: PREPARE THE IMAGE ---
# Same processing as before
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

image = Image.open(image_path).convert('RGB') # Open image
input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension -> [1, 3, 128, 128]

# --- STEP 5: ASK THE AI ---
with torch.no_grad(): # No need to calculate gradients for testing
    output = model(input_tensor)
    _, predicted_index = torch.max(output, 1)
    
    # Map index 0 -> 'no', 1 -> 'yes' (Alphabetical order)
    class_names = ['no', 'yes']
    prediction = class_names[predicted_index.item()]

# --- STEP 6: REPORT CARD ---
print(f"AI Prediction:  {prediction.upper()}")

if prediction == true_label:
    print("✅ The AI was CORRECT!")
else:
    print("❌ The AI was WRONG.")