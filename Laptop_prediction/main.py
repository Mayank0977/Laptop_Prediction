import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import base64

# Correct path handling
BASE_DIR = os.path.dirname(__file__)

pipe_path = os.path.join(BASE_DIR, "pipe.pkl")
df_path = os.path.join(BASE_DIR, "df.pkl")

pipe = pickle.load(open(pipe_path, 'rb'))
df = pickle.load(open(df_path, 'rb'))

st.title("Laptop Price Prediction")

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    background = f'''
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(background, unsafe_allow_html=True)

# Call the function
set_background(os.path.join(BASE_DIR, "laptop.jpg"))

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type
Type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram (in GB)', [8,16,32,64,128,256,512])

# Weight
weight = st.number_input('Weight of the laptop', min_value=1)

# TouchScreen
touchscreen = st.selectbox("Touchscreen", ['Yes', 'No'])

# IPS
ips = st.selectbox("IPS", ['Yes', 'No'])

# Screen size
screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox("Resolution", [
    '1920x1080','1366x768','1600x900','3840x2160',
    '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
])

# CPU
cpu = st.selectbox("CPU", df['Cpu brand'].unique())

# HDD
hdd = st.selectbox("HDD (in GB)", [0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox("SSD (in GB)", [0,8,16,32,64,128,256,512,1024])

# GPU
gpu = st.selectbox("GPU", df['Gpu brand'].unique())

# OS
os_choice = st.selectbox("OS", df['os'].unique())

if st.button('Predict Price'):
    # Convert inputs
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Build query
    query = pd.DataFrame([[company, Type, ram, os_choice, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu]],
                         columns=['Company', 'TypeName', 'Ram', 'os', 'Weight',
                                  'Touchscreen', 'IPS', 'ppi', 'Cpu brand', 'HDD', 'SSD', 'Gpu brand'])

    st.title("The predicted price of this configuration is ₹" +
             str(int(np.exp(pipe.predict(query)[0]))))
