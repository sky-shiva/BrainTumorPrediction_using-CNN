import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# --- STEP 1: CHECK FOR YOUR GPU ---
# This line checks if your RTX 2050 is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("Your RTX 2050 is ready to train!")
else:
    print("WARNING: Running on CPU. Check your PyTorch installation.")

# --- STEP 2: PREPARE DATA ---
# Resize images to 128x128
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load data from the 'Data' folder
# Make sure you have 'Data/yes' and 'Data/no' inside your VS Code folder
if os.path.exists('Data'):
    dataset = datasets.ImageFolder(root='Data', transform=transform)
    
    # Split: 80% Train, 20% Test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    # Loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    print(f"Found {len(dataset)} images. Classes: {dataset.classes}")
else:
    print("ERROR: Could not find 'Data' folder. Please create 'Data' and put 'yes'/'no' folders inside.")
    exit()

# --- STEP 3: BUILD MODEL ---
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__() # this helps to inherit nn.Module features , basically it starts the nn.Module
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Layer 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Flatten: 128 channels * 16 * 16 size (after 3 pools of 128x128)
        self.fc1 = nn.Linear(128 * 16 * 16, 2) 

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16) # Flatten
        x = self.fc1(x)
        return x

# SEND MODEL TO GPU
model = BrainTumorCNN().to(device)

# --- STEP 4: TRAIN ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nStarting Training...")

for epoch in range(10): # 10 Epochs
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        # CRITICAL: Send images/labels to GPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy just for fun
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {100 * correct / total:.2f}%")

print("Training Complete!")

# --- STEP 5: SAVE THE MODEL ---
torch.save(model.state_dict(), "brain_tumor_model.pth")
print("Model saved as 'brain_tumor_model.pth'")