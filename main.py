import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    model = torch.load("model_2_ver_3.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

# PyTorch Model Prediction
def model_prediction(test_image):
    model = load_model()

    image = Image.open(test_image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Adjust to your model's normalization
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
    return predicted

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.header("CHEST X-RAY DIAGNOSIS SYSTEM 🔍")
    st.markdown("Our mission is to help in Detect chest diseases efficiently. Upload an image of x-ray image, and our system will analyze it to detect any signs of diseases.")
    image_path = "home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    

    ### How It Works
    1. **Upload Image:** Go to the **Disease detection** page and upload the Image.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action or Reports.
    """)

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    *******************************************************************************************************************************
    This dataset consists of about 120k x-ray images of healthy, anomalous and diseased chest x-ray images categorized into 14 different classes.
    #### Content
    1. train (80k images)
    2. test (20k images)
    3. validation (20k images)
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image") and test_image is not None:
        st.image(test_image, use_container_width=True)

    if st.button("Predict") and test_image is not None:
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        # Replace with your actual disease classes
        class_name = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia'
        ]

        st.success("Model is Predicting it's a case of: **{}**".format(class_name[result_index]))
