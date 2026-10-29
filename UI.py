import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
from fpdf import FPDF


# =====================================================================
# 1. MODEL ARCHITECTURE CLASSES
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


class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.3402)  # Using an estimated calibrated temp

    def forward(self, input):
        logits = self.model(input)
        return logits / self.temperature


# =====================================================================
# 2. GRAD-CAM UTILITY
# =====================================================================
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
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().detach().numpy()


def get_gradcam_image(model, img_tensor, original_image_pil):
    target_layer = model.model.cbam
    grad_cam = GradCAM(model.model, target_layer)
    heatmap = grad_cam.generate_heatmap(img_tensor.unsqueeze(0))

    img = np.array(original_image_pil.resize((224, 224)))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)
    return Image.fromarray(overlay)


# =====================================================================
# 3. PDF REPORT GENERATOR
# =====================================================================
def generate_pdf_report(original_img, gradcam_img, pred_class, confidence, status):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    # Header
    pdf.cell(200, 10, txt="Fetal Ultrasound Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Result Details
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Plane: {pred_class}", ln=True)
    pdf.cell(200, 10, txt=f"Calibrated Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"Selective Prediction Status: {status}", ln=True)
    pdf.ln(10)

    # Save temp images to embed in PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_orig:
        original_img.save(tmp_orig.name)
        orig_path = tmp_orig.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_grad:
        gradcam_img.save(tmp_grad.name)
        grad_path = tmp_grad.name

    # Embed Images
    pdf.cell(200, 10, txt="Original Ultrasound:", ln=True)
    pdf.image(orig_path, x=10, y=None, w=90)
    pdf.ln(5)

    pdf.cell(200, 10, txt="Grad-CAM (Explainable AI Heatmap):", ln=True)
    pdf.image(grad_path, x=10, y=None, w=90)

    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')


# =====================================================================
# 4. STREAMLIT APP LOGIC
# =====================================================================
st.set_page_config(page_title="Fetal Plane Classifier", layout="wide")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = ProposedFetalModelDenseNet(num_classes=6)
    # Load your trained weights (Ensure this file is in the same folder)
    base_model.load_state_dict(torch.load("best_densenet_fetal.pth", map_location=device))
    calibrated_model = ModelWithTemperature(base_model)
    calibrated_model.to(device)
    calibrated_model.eval()
    return calibrated_model, device


CLASS_NAMES = ['Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other']

# Initialize App
st.title("🩺 Fetal Ultrasound Plane Classification")
st.markdown(
    "Upload a fetal ultrasound to classify the anatomical plane using a **DenseNet-121 + CBAM** architecture. Features integrated **Temperature Calibration** and **Selective Prediction**.")

# Sidebar Configurations
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Selective Prediction Threshold (%)", min_value=50, max_value=100, value=85, step=1)

# Load Model
try:
    model, device = load_model()
except Exception as e:
    st.error(f"Failed to load model weights. Make sure 'best_densenet_fetal.pth' is in the directory. Error: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an ultrasound image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read Image
    original_pil = Image.open(uploaded_file).convert('L')

    # Display Layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(original_pil, use_container_width=True)

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = preprocess(original_pil).to(device)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    conf_value = conf.item() * 100
    pred_class = CLASS_NAMES[pred_idx.item()]

    # Selective Prediction Logic
    if conf_value >= threshold:
        ui_status = "✅ ACCEPTED (Automated Prediction)"
        pdf_status = "ACCEPTED (Automated Prediction)"
        color = "green"
    else:
        ui_status = "⚠️ REVIEW (Route to Human Expert)"
        pdf_status = "REVIEW (Route to Human Expert)"
        color = "red"

    with col2:
        st.subheader("Grad-CAM Explainability")
        with st.spinner("Generating heatmap..."):
            gradcam_pil = get_gradcam_image(model, input_tensor, original_pil)
        st.image(gradcam_pil, use_container_width=True)

    with col3:
        st.subheader("Analysis Results")
        st.markdown(f"**Predicted Plane:** {pred_class}")
        st.markdown(f"**Confidence:** {conf_value:.2f}%")
        st.markdown(f"**Recommendation:** <span style='color:{color}; font-weight:bold;'>{ui_status}</span>",
                    unsafe_allow_html=True)

        # Generate PDF using the clean text without emojis
        pdf_bytes = generate_pdf_report(original_pil, gradcam_pil, pred_class, conf_value, pdf_status)

        st.download_button(
            label="📄 Download Medical Report",
            data=pdf_bytes,
            file_name="fetal_ultrasound_report.pdf",
            mime="application/pdf"
        )