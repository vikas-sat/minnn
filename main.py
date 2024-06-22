from PIL import Image  
import streamlit as st
from ultralytics import YOLO
import google.generativeai as genai

model_path = 'C:/Users/vikas/OneDrive/Desktop/javalabss/best (3).pt'
info = "Basic information on"

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Image/Video Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = 0.4

# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
if source_img:
    with col1:
        # Opening the uploaded image
        uploaded_image = Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(uploaded_image,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

# Load the model
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Detect objects if the button is clicked
if st.sidebar.button('Detect Objects') and source_img:
    try:
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        with col2:
            st.image(res_plotted,
                     caption='Detected Image',
                     use_column_width=True
                     )
            try:
                with st.expander("Detection Results"):
                    if boxes:
                        for box in boxes:
                            st.write(box.xywh)
                            c = box.cls
                            nam = model.names[int(c)]
                            info += " " + nam
                    else:
                        st.write("Insufficient data: No objects detected.")
            except Exception as ex:
                st.write("Error displaying detection results.")
                st.error(ex)
    except Exception as ex:
        st.write("Error during object detection.")
        st.error(ex)

# Placeholder for Google Generative AI integration
# Ensure to use the correct package and API for Google services
# Here is an example with a placeholder text:
# # Integrate Google Generative AI for additional information
API_KEY = 'AIzaSyCCZcRJtxfou0Y3Tx0GIFAsu5UOolbAwgw'  # Replace with your actual API key
genai.configure(api_key=API_KEY)
geminiModel = genai.GenerativeModel("gemini-pro")

try:
    response = geminiModel.generate_content(info)
    st.write("**Additional Information:**")
    st.write(response.text)
except Exception as ex:
    st.write("Error generating additional information.")
    st.error(ex)
