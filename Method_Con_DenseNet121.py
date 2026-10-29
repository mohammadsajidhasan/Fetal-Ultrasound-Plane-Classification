import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import cv2
import os

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
ROOT_DIR = r"G:\UIU FILES\12th Trimester\DIP\DIP Project\FETAL_PLANES_ZENODO"
CSV_PATH = os.path.join(ROOT_DIR, "FETAL_PLANES_DB_data.csv")
IMG_DIR = os.path.join(ROOT_DIR, "Images")

IMAGE_COL_NAME = "Image_name"
LABEL_COL_NAME = "Plane"
BEST_WEIGHTS_PATH = "best_densenet_fetal.pth"


# =====================================================================
# 2. MODEL & DATASET DEFINITIONS (Required to load weights properly)
# =====================================================================
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
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ProposedFetalModelDenseNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ProposedFetalModelDenseNet, self).__init__()
        self.backbone = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.backbone.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = 1024

        self.features = self.backbone.features
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(in_features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mc_dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.mc_dropout(x)
        x = self.classifier(x)
        return x


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
        image = Image.open(img_path).convert('L')
        label = self.class_to_idx[self.data.iloc[idx][LABEL_COL_NAME]]
        if self.transform:
            image = self.transform(image)
        return image, label


# =====================================================================
# 3. EVALUATION PIPELINE METHODS
# =====================================================================

class ModelWithTemperature(nn.Module):
    """Applies Temperature Scaling to calibrate model probabilities."""

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature


def calibrate_model(calibrated_model, val_loader, device):
    calibrated_model.to(device)
    nll_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.LBFGS([calibrated_model.temperature], lr=0.01, max_iter=50)

    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = calibrated_model.model(images)
            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(calibrated_model.temperature_scale(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)
    print(f"\n[Step 9] Model Calibrated. Optimal Temperature: {calibrated_model.temperature.item():.4f}")
    return calibrated_model


def selective_prediction(calibrated_model, dataloader, device, threshold=0.85):
    calibrated_model.eval()
    all_labels, all_preds, all_confs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = calibrated_model(images)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_confs = np.array(all_confs)

    # Apply Threshold Logic
    accepted_mask = all_confs >= threshold
    rejected_mask = ~accepted_mask

    # Metrics for OVERALL Test Set
    print("\n--- OVERALL TEST SET METRICS (Before Rejection) ---")
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds) * 100:.2f}%")
    print(f"Precision: {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")

    # Metrics for ACCEPTED Predictions Only
    coverage = np.mean(accepted_mask) * 100
    acc_labels = all_labels[accepted_mask]
    acc_preds = all_preds[accepted_mask]

    print(f"\n[Step 10] Selective Prediction (Threshold: {threshold})")
    print(f"Total Samples Processed: {len(all_labels)}")
    print(f"REJECTED (To Human Expert): {np.sum(rejected_mask)} ({100 - coverage:.1f}% routed for review)")
    print(f"ACCEPTED (Automated):       {np.sum(accepted_mask)} ({coverage:.1f}% coverage)")
    print(f"\n--- METRICS FOR ACCEPTED PREDICTIONS ONLY ---")
    print(f"Accuracy:  {accuracy_score(acc_labels, acc_preds) * 100:.2f}%")
    print(f"Precision: {precision_score(acc_labels, acc_preds, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(acc_labels, acc_preds, average='macro', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(acc_labels, acc_preds, average='macro', zero_division=0):.4f}")

    return all_labels, all_preds


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward(retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)
        return heatmap.cpu().detach().numpy()


def visualize_gradcam(model, image_tensor, original_image_path):
    target_layer = model.model.cbam
    grad_cam = GradCAM(model.model, target_layer)
    image_tensor = image_tensor.unsqueeze(0)

    heatmap = grad_cam.generate_heatmap(image_tensor)

    img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_colored, 0.5, 0)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Ultrasound")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Grad-CAM Overlay")
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()
    print("\n[Step 11] Grad-CAM visualization generated. (Close image window to end script)")


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'})
    plt.title("Confusion Matrix (Error Analysis)")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("\n[Step 12] Confusion Matrix generated. (Close image window to continue)")


# =========================================================
# 4. MAIN EXECUTION SEQUENCE
# =========================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Evaluation Pipeline on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("Loading Dataset to recreate exact splits...")
    full_dataset = FetalCSVDataset(csv_file=CSV_PATH, img_dir=IMG_DIR, transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Manual seed ensures the test set matches exactly what it was during training
    _, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Loading Model Weights from: {BEST_WEIGHTS_PATH} ...")
    model = ProposedFetalModelDenseNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, weights_only=True))

    print("\n=======================================================")
    print("        STARTING POST-TRAINING EVALUATION STEPS        ")
    print("=======================================================")

    # Step 1: Wrap your trained DenseNet model with Temperature Scaling
    calibrated_model = ModelWithTemperature(model)

    # Step 2 (Step 9 in Diagram): Calibrate on Validation Set
    calibrated_model = calibrate_model(calibrated_model, val_loader, device)

    # Step 3 (Step 10 in Diagram): Run Selective Prediction and Print Metrics
    labels, preds = selective_prediction(calibrated_model, test_loader, device, threshold=0.85)

    # Step 4 (Step 12 in Diagram): Generate Error Analysis / Confusion Matrix
    plot_confusion_matrix(labels, preds, class_names)

    # Step 5 (Step 11 in Diagram): Generate a Grad-CAM example for a single image
    # We grab the first image from the test set for the demonstration
    actual_idx = test_dataset.indices[0]
    sample_image_tensor, _ = test_dataset[0]

    img_name = str(full_dataset.data.iloc[actual_idx][IMAGE_COL_NAME])
    if not img_name.lower().endswith('.png'):
        img_name += '.png'
    sample_image_path = os.path.join(IMG_DIR, img_name)

    visualize_gradcam(calibrated_model, sample_image_tensor.to(device), sample_image_path)