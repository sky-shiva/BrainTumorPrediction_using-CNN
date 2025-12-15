import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ==========================================
# 👇 TYPE YOUR IMAGE NAME HERE 👇
# ==========================================
image_name = "15 no.jpg"  # <--- CHANGE THIS to your file name
# ==========================================

# --- STEP 1: DEFINE THE MODEL ARCHITECTURE ---
# (Must match the training script exactly)
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

# --- STEP 2: LOAD THE TRAINED BRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)

if os.path.exists("brain_tumor_model.pth"):
    model.load_state_dict(torch.load("brain_tumor_model.pth"))
    model.eval() # Set to evaluation mode
    print("✅ Model loaded successfully!")
else:
    print("❌ Error: 'brain_tumor_model.pth' not found. Did you run train.py?")
    exit()

# --- STEP 3: PREPARE YOUR IMAGE ---
if not os.path.exists(image_name):
    print(f"❌ Error: Could not find image '{image_name}'")
    print("👉 Make sure the image is in the same folder as this script.")
    exit()

# Same preprocessing as training (Resize -> Tensor -> Normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

try:
    image = Image.open(image_name).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
except Exception as e:
    print(f"❌ Error opening image: {e}")
    exit()

# --- STEP 4: PREDICT ---
print(f"\n🔍 Analyzing {image_name}...")

with torch.no_grad():
    output = model(input_tensor)
    
    # Calculate probabilities (Confidence)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)
    
    # Map index to class name
    # Index 0 is 'no' (Healthy), Index 1 is 'yes' (Tumor)
    class_names = ['No Tumor (Healthy)', 'Yes (Brain Tumor)']
    result = class_names[predicted_index.item()]
    percentage = confidence.item() * 100

# --- STEP 5: SHOW RESULT ---
print("-" * 30)
print(f"RESULT: {result.upper()}")
print(f"CONFIDENCE: {percentage:.2f}%")
print("-" * 30)

if predicted_index.item() == 1:
    print("⚠️  Recommendation: Consult a Doctor.")
else:
    print("✅  Clean Bill of Health.")