import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_image_comparison import image_comparison
import torch

# Importing the RRDBNet architecture from the ESRGAN project
import RRDBNet_arch as arch 

from accelerate import Accelerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

accelerator = Accelerator()

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

import pytesseract
import easyocr
import requests
import os
import sys
import gdown

TF_ENABLE_ONEDNN_OPTS=0


def download_model(url, dest_path):
    """Download the classification model from gdrive"""
    # response = requests.get(url)
    # with open(dest_path, 'wb') as f:
    #     f.write(response.content)

    gdown.download(url, dest_path)

def tesseract_read(image):
    custom_config = r'--psm 6'
    return pytesseract.image_to_string(image, config = custom_config)

def easyocr_read(image):
    reader = easyocr.Reader(lang = ["id"], gpu = False)
    return reader.readText(image, detail = 0)

@st.cache_data
def prepare_enhance_model(esrgan_model_gdrive):
    """load the ESRGAN model

    esrgan_model_path is the sharing link of the model on google drive
    """ 

    recog_model_file_name = "esrgan.pth"

    if not os.path.exists("enhance_models"):  # make new dir if it does not exist
        os.makedirs("enhance_models")
    esrgan_model_path = os.path.join("enhance_models", recog_model_file_name)
    
    download_model(esrgan_model_gdrive, esrgan_model_path)

    enhance_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    enhance_model.load_state_dict(torch.load(esrgan_model_path), strict=True)
    enhance_model.eval()
    return enhance_model

@st.cache_resource
def select_recog_model(recog_model_selected):

    if recog_model_selected == "ResNet50":
        recog_model_path = r"https://drive.google.com/uc?id=1J9llUuzdCRtPmqEjzI5fPjwbCj0XDdSt/view?usp=sharing"
        recog_model_file_name = "ResNet50.keras"
    elif recog_model_selected == "VGG16":
        recog_model_path = r" https://drive.google.com/uc?id=1vqk2ehrci4M_DwSL7aAt8z-wYB7bonkI/view?usp=drive_link"
        recog_model_file_name = "VGG16.keras"
 
    if not os.path.exists("recog_models"):  # make new dir if it does not exist
        os.makedirs("recog_models")

    model_path = os.path.join("recog_models", recog_model_file_name)
    download_model(recog_model_path, model_path)
    return tf.keras.models.load_model(model_path)

with st.sidebar:
    selected = option_menu("Main Menu", ["File Upload","Recognition", 'Enhancement', "OCR"], 
        icons=['cloud-upload','house', 'gear', "text"], menu_icon="cast", default_index=0)

# resize function
def pad_and_resize(_image, target_size):
    # old_size = _image.size  # (width, height)
    # ratio = float(target_size[1]) / max(old_size)
    # new_size = tuple([int(x * ratio) for x in old_size])
    # _image = _image.resize(new_size, Image.LANCZOS)
    
    # new_image = Image.new("RGB", target_size, (255, 255, 255))
    # new_image.paste(_image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    new_image = _image.resize(target_size)

    return new_image

# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'classify_model' not in st.session_state:
    st.session_state['classify_model'] = None
if 'enhanced_file' not in st.session_state:
    st.session_state['enhanced_file'] = None
if 'show_enhance_option' not in st.session_state:
    st.session_state['show_enhance_option'] = False
if 'show_ocr_option' not in st.session_state:
    st.session_state['show_ocr_option'] = False

if selected == "File Upload":
    st.title("Upload File")
    st.divider()
    new_file = st.file_uploader("Insert an image!", type = "jpg") 
    if new_file is not None:
    # st.session_state['classify_model'] = tf.keras.models.load_model("recog_KTP\\model\\best_model.keras")
        st.session_state['classify_model'] = None
        st.session_state['uploaded_file'] = new_file
        st.session_state['enhanced_file'] = None
        st.session_state['show_ocr_option'] = False
        st.session_state['show_enhance_option'] = False

if selected == "Recognition":
    st.title("KTP recognition")
    st.divider()
    st.write("This is a simple classification model to identify whether a document is an Indonesian Government ID")

    recog_model_selected = st.radio("Pick a model", ["ResNet50", "VGG16"])
    st.session_state['classify_model'] = select_recog_model(recog_model_selected)
    classify_button_clicked = st.button("classify")

    if classify_button_clicked:
        if st.session_state['uploaded_file'] is not None:

            image = Image.open(st.session_state['uploaded_file'])
            print(np.array(image))
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            if recog_model_selected == "ResNet50_1" or recog_model_selected == "ResNet50_2":
                target_size = (256, 256)
            else:
                target_size = (539, 856)

            image = pad_and_resize(image, target_size)
            print("after padding and resizing :", np.array(image))
            image = np.array(image) / 255.0
            print("after normalization :", image)
            image = np.expand_dims(image, axis = 0)
            print("after image expanding", image)
            # image = preprocess_image(image, target_size)

            # datagen = ImageDataGenerator(rescale=1./255)

            # image = Image.open(st.session_state['uploaded_file'])
            # image.save(r"input/input_image.jpg")
            # generator = datagen.flow_from_directory(
            #     "input",  # Training directory path
            #     target_size=(256, 256),  # Adjust based on your model's input size
            #     class_mode='binary',
            #     shuffle=False
            # )

            with st.spinner("Classifying ..."):
                # run the model on the image input
                prediction = st.session_state['classify_model'].predict(image)
                print(prediction)

            threshold = 0.5
            predicted_class = "Indonesian ID" if prediction < threshold else "Other Document"
            if predicted_class == "Indonesian ID":
                st.session_state["show_enhance_option"] = True
                st.session_state["show_ocr_option"] = True
            
            st.write(f'The predicted class is {predicted_class}')
        else:
            st.write("Image is not found, please insert an image on the designated page!")

if selected == "Enhancement":
    st.title("Image Enhancement")
    enhance_button_clicked = st.button("Enhance")
    st.divider()
    if enhance_button_clicked:
        if st.session_state["show_enhance_option"]:
            device = accelerator.device

            model_path = r"https://drive.google.com/uc?id=1Jt8scnQz0Bk548zGUbtnUeIOgkMFjrbl/view?usp=sharing"
            
            enhance_model = prepare_enhance_model(model_path)

            original_file = Image.open(st.session_state['uploaded_file']).convert("RGB")
            uploaded_file = np.array(original_file) / 255.0
            uploaded_file= torch.from_numpy(np.transpose(uploaded_file, (2, 0, 1))).float()
            uploaded_file = uploaded_file.unsqueeze(0)
            img_LR = uploaded_file.to(device)

            with torch.no_grad():
                output = enhance_model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            output = Image.fromarray(output)

            # render image-comparison
            image_comparison(
                img1=original_file,
                img2=output,
                label1="Original",
                label2="Enhanced",
                show_labels = True
            )
            st.session_state["enhanced_file"] = output
        else:
            st.write("Image is not found, please insert an image on the designated page!")

if selected == "OCR":
    st.title("Image OCR")
    st.divider()
    
    if st.session_state["show_ocr_option"]:
        ocr_model = st.selectbox("Select OCR reader", ["Tesseract", "Easyocr"])

        if ocr_model == "Tesseract":
            ocr_reader = tesseract_read
        elif ocr_model == "Easyocr":
            ocr_reader = easyocr_read
        
        text_output = None

        input_image = st.selectbox("Select input image", ["Original", "Enhanced"])
        if input_image == "Enhanced":
            if st.session_state["enhanced_file"] == None:
                st.error("The file has not been enhanced")
            else:
                enhanced_image = Image.open(st.session_state["enhanced_file"])
                text_output = ocr_reader(enhanced_image)
        elif input_image == "Original":
            original_image = Image.open(st.session_state["uploaded_file"])
            text_output = ocr_reader(original_image)
            
        NIK_text = None
        if text_output:
            for i in range(len(text_output)):
                if text_output[i] == "NIK" and len(text_output) >= i:
                    NIK_text = text_output[i+1]
        st.write(NIK_text)

    else:
        st.write("Image is not found, please insert an image on the designated page!")

