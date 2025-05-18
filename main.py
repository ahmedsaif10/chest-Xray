import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define your model class (same architecture used in training)
class ChestXRayModel(nn.Module):
    def __init__(self):
        super(ChestXRayModel, self).__init__()
        # Example structure (replace with your actual architecture)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 64 * 64, 8)  # assuming 8 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the PyTorch model
def load_model():
    model = ChestXRayModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Make prediction
def model_prediction(image, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

# List of disease classes
class_name = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
              'Effusion', 'Emphysema', 'Fibrosis', 'Hernia']

# Streamlit App
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

if app_mode == "Home":
    st.header("CHEST X-RAY DIAGNOSIS SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("Welcome to our chest X-ray diagnosis system!")

elif app_mode == "About":
    st.header("About")
    st.markdown("Chest X-ray dataset with 8 disease classes.")

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        if st.button("Predict"):
            model = load_model()
            result_index = model_prediction(image, model)
            st.success(f"The model predicts: **{class_name[result_index]}**")

