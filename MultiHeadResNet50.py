import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, datasets, transforms

# ===============================
# Hyperparameters
# ===============================

BATCH_SIZE = 64
NUM_CLASSES = 10          # CIFAR-10
LR = 1e-3
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128  # smaller than 224 for speed

# ===============================
# Multi-Head Model (ResNet18)
# ===============================
class MultiHeadResNet18(nn.Module):
    def __init__(self, num_classes_classification, include_extra_heads=False):
        super(MultiHeadResNet18, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes_classification)
        )
        
        self.include_extra_heads = include_extra_heads
        if include_extra_heads:
            # Binary head example
            self.binary_head = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        features = self.backbone(x)
        class_out = self.classification_head(features)
        if self.include_extra_heads:
            binary_out = self.binary_head(features)
            return class_out, binary_out
        else:
            return class_out

# ===============================
# Main function
# ===============================
def main():
    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    
    # CIFAR-10 dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transforms  # change transform for val

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Initialize model with extra head
    model = MultiHeadResNet18(NUM_CLASSES, include_extra_heads=True).to(DEVICE)

    # Loss & optimizer
    criterion_class = nn.CrossEntropyLoss()
    criterion_binary = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Binary labels dummy: 1 אם class זוגי, 0 אם אי-זוגי
            binary_labels = (labels % 2).float().unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            class_out, binary_out = model(imgs)
            
            loss_class = criterion_class(class_out, labels)
            loss_binary = criterion_binary(binary_out, binary_labels)
            loss = loss_class + loss_binary  # סך כל ה-loss
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        binary_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                binary_labels = (labels % 2).float().unsqueeze(1).to(DEVICE)
                
                class_out, binary_out = model(imgs)
                
                loss_class = criterion_class(class_out, labels)
                loss_binary = criterion_binary(binary_out, binary_labels)
                val_loss += (loss_class + loss_binary).item()
                
                # Classification accuracy
                _, predicted = torch.max(class_out, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Binary head accuracy
                binary_pred = (binary_out > 0.5).float()
                binary_correct += (binary_pred == binary_labels).sum().item()
        
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, "
              f"Class Acc: {correct/total:.4f}, Binary Acc: {binary_correct/total:.4f}\n")


# ===============================
# Run main
# ===============================
if __name__ == "__main__":
    main()
