import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from PIL import Image
import os

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
ROOT_DIR = r"G:\UIU FILES\12th Trimester\DIP\DIP Project\FETAL_PLANES_ZENODO"
CSV_PATH = os.path.join(ROOT_DIR, "FETAL_PLANES_DB_data.csv")
IMG_DIR = os.path.join(ROOT_DIR, "Images")

IMAGE_COL_NAME = "Image_name"
LABEL_COL_NAME = "Plane"


# =====================================================================

# ---------------------------------------------------------
# 2. CBAM ATTENTION BLOCKS (From Diagram)
# ---------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Applies Channel Attention then Spatial Attention
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ---------------------------------------------------------
# 3. PROPOSED MODEL ARCHITECTURE (DenseNet-121)
# ---------------------------------------------------------
class ProposedFetalModelDenseNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ProposedFetalModelDenseNet, self).__init__()
        # 1. Backbone: DenseNet-121
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        # Adapt for 1-channel Grayscale input
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # DenseNet-121 outputs 1024 channels
        in_features = 1024

        self.features = self.backbone.features
        self.relu = nn.ReLU(inplace=True)

        # 2. CBAM (Channel + Spatial Attention)
        self.cbam = CBAM(in_features)

        # 3. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 4. MC Dropout (Uncertainty)
        self.mc_dropout = nn.Dropout(p=0.5)

        # 5. Classification Head
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Base Feature Extraction
        x = self.features(x)
        x = self.relu(x)

        # Step 1: CBAM (Channel + Spatial Attention)
        x = self.cbam(x)

        # Step 2: Global Average Pooling
        x = self.global_pool(x)

        # Step 3: Feature Vector (Flattening)
        x = torch.flatten(x, 1)

        # Step 4: MC Dropout (Uncertainty)
        x = self.mc_dropout(x)

        # Step 5 & 6: Classification Head -> Output (6 Classes)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# 4. DATASET LOADER
# ---------------------------------------------------------
class FetalCSVDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file, sep=';')
        if IMAGE_COL_NAME not in self.data.columns:
            self.data = pd.read_csv(csv_file, sep=',')

        self.classes = sorted(self.data[LABEL_COL_NAME].unique().tolist())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = str(self.data.iloc[idx][IMAGE_COL_NAME])
        if not img_name.lower().endswith('.png'):
            img_name += '.png'

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to 1-Channel Grayscale

        label_str = self.data.iloc[idx][LABEL_COL_NAME]
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------------
# 5. EARLY STOPPING UTILITY (Two-Strike System)
# ---------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, path="best_model.pth"):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)


def run_test_evaluation(model, test_loader, device, weights_path="best_model.pth"):
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


# =========================================================
# 6. MAIN EXECUTION
# =========================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("Loading dataset from CSV...")
    full_dataset = FetalCSVDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform)
    num_classes = len(full_dataset.classes)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize the updated DenseNet-121 Diagram Model
    model = ProposedFetalModelDenseNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    early_stopping = EarlyStopping(patience=5)

    MAX_EPOCHS = 30
    strikes = 0
    MAX_STRIKES = 2
    BEST_WEIGHTS = "best_densenet_fetal.pth"

    print("\n--- Starting Training ---")
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{MAX_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model, path=BEST_WEIGHTS)
        if early_stopping.early_stop:
            strikes += 1
            print(f"\n[!] EARLY STOPPING TRIGGERED (Strike {strikes}/{MAX_STRIKES})")
            print("Validation loss hasn't improved. Pausing to evaluate test metrics on current best model...")

            run_test_evaluation(model, test_loader, device, weights_path=BEST_WEIGHTS)

            if strikes >= MAX_STRIKES:
                print("\nNo improvement happened again. Permanently halting training to prevent overfitting.")
                break
            else:
                print("\nResuming training for remaining epochs...")
                early_stopping.early_stop = False
                early_stopping.counter = 0
                model.load_state_dict(torch.load(BEST_WEIGHTS, weights_only=True))

    print("\n--- Final Complete Evaluation on Test Set ---")
    run_test_evaluation(model, test_loader, device, weights_path=BEST_WEIGHTS)