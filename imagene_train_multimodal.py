import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd, os, numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

# ----- DATASET -----
class IDCDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.id + ".png")
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        gender = 1 if row.gender == "M" else 0
        clin = torch.tensor([row.age/100.0, gender, row.smoker], dtype=torch.float32)
        label = torch.tensor(row.label, dtype=torch.long)
        return image, clin, label

# ----- TRANSFORMS -----
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# ----- SPLIT -----
df = pd.read_csv("data/clinical.csv")
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)

train_ds = IDCDataset("data/train.csv", "data/images", transform)
val_ds = IDCDataset("data/val.csv", "data/images", transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ----- MODEL -----
class MultiModalNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        base.fc = nn.Identity()
        self.cnn = base
        self.img_fc = nn.Linear(512, 128)

        self.clin_fc = nn.Sequential(nn.Linear(3,16), nn.ReLU(), nn.Linear(16,32))

        self.classifier = nn.Sequential(nn.Linear(160,64), nn.ReLU(), nn.Linear(64,2))

    def forward(self, img, clin):
        x_img = F.relu(self.img_fc(self.cnn(img)))
        x_clin = self.clin_fc(clin)
        x = torch.cat([x_img, x_clin], dim=1)
        return self.classifier(x)

model = MultiModalNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ----- TRAIN LOOP -----
for epoch in range(3):
    model.train(); total_loss = 0
    for img, clin, y in train_loader:
        img, clin, y = img.to(device), clin.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(img, clin)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train loss = {total_loss/len(train_loader):.4f}")

# ----- EVALUATION -----
model.eval(); ys, ps = [], []
with torch.no_grad():
    for img, clin, y in val_loader:
        img, clin = img.to(device), clin.to(device)
        logits = model(img, clin)
        prob = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        ys.extend(y.numpy()); ps.extend(prob)

acc = accuracy_score(ys, np.round(ps))
auc = roc_auc_score(ys, ps)
print(f"✅ Validation Accuracy: {acc:.3f}  AUROC: {auc:.3f}")
