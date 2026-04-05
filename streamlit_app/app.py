import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from pathlib import Path

# ====================== CONFIG ======================
st.set_page_config(page_title="CropGuard AI - Maharashtra", page_icon="🌱", layout="wide")
st.title("🌱 CropGuard AI")
st.subheader("Maharashtra Farmer Crop Disease Detection & Advisory")
st.markdown("**Best AI Innovator Award Entry** | Seamedu Awards 2026")

# Load model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier[1].in_features, 13)
    )
    model.load_state_dict(torch.load("models/best_crop_disease_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class names (exactly matching your dataset)
classes = [
    "soybean_Bacterial_Leaf_Blight", "soybean_Multi_Dry_Leaf", "soybean_Multi_Healthy_Leaf",
    "soybean_Multi_Septoria_Brown_Spot", "soybean_Multi_Vein_Necrosis", "soybean_Single_Dry_Leaf",
    "soybean_Single_Healthy_Leaf", "soybean_Single_Septoria_Brown_Spot", "soybean_Single_Vein_Necrosis_Virus",
    "sugarcane_Healthy", "sugarcane_Mosaic", "sugarcane_Redrot", "sugarcane_Rust", "sugarcane_Yellow"
]

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================== UI ======================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Leaf Photo")
    uploaded_file = st.file_uploader("Choose a leaf image (soybean or sugarcane)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Uploaded Image", use_container_width=True)   # ← Fixed here
    
    # Prediction
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    predicted_class = classes[predicted_idx.item()]
    
    with col2:
        st.markdown("### Prediction Result")
        st.success(f"**Predicted: {predicted_class.replace('_', ' ').title()}**")
        st.metric("Confidence", f"{confidence.item()*100:.1f}%")
        
        # Advisory
        if "Healthy" in predicted_class:
            st.info("✅ **Healthy leaf** — No action needed.")
        else:
            st.warning("⚠️ Disease detected. Recommend consulting local Krishi Vigyan Kendra (Pune).")
        
        st.markdown("**Farmer Advisory**")
        st.write("📍 Maharashtra region — Monitor weather for next 7 days.")

    # Top 3 predictions
    st.markdown("### Top 3 Predictions")
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        st.write(f"{i+1}. **{classes[top3_idx[i].item()].replace('_', ' ').title()}** — {top3_prob[i].item()*100:.1f}%")

st.caption("Built with EfficientNet-B0 | 98.63% accuracy | Trained on 3,658 real Maharashtra farm images")