import numpy as np
import streamlit as st
import os 
import imageio 
import tensorflow as tf 
from PIL import Image
from utils import load_data, num_to_char
from modelutil import load_model
import Levenshtein
import matplotlib.pyplot as plt

# Function to calculate character-level accuracy
def calculate_accuracy(true_text, predicted_text):
    correct_chars = sum(1 for true_char, pred_char in zip(true_text, predicted_text) if true_char == pred_char)
    total_chars = max(len(true_text), len(predicted_text))
    accuracy = correct_chars / total_chars * 100
    return accuracy

# Function to calculate word error rate (WER)
def calculate_wer(true_text, predicted_text):
    return Levenshtein.distance(true_text.split(), predicted_text.split()) / len(true_text.split()) * 100

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Inject custom CSS to change cursor to hand for options
st.markdown("""
    <style>
        .sidebar-content ul {
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.image('https://lh3.googleusercontent.com/p/AF1QipMBXyOUmcjkumz9Zb4CqXIrmOl51PPKJ0TnxRka=s1360-w1360-h1020')
    st.title('LipBuddy')
    st.info('Developed by:')
    st.subheader('V. Sai Tarun')
    st.subheader('P. Chandan Lohit')
    st.subheader('P. Sailaja Devi')
    st.subheader('P. Yeshwanth Sai')

st.title('LipNet App') 
# Generating a list of options or videos 
options = os.listdir("C:\PROJECT\LipNet\data\s1")
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 
    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        video_data, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # Convert the video data to uint8 and reshape if needed
        video_data = video_data.numpy().astype('uint8')
        if video_data.shape[0] == 1:
            video_data = video_data.squeeze(axis=0)
        

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_data, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Output')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

        # Load true annotations
        true_text = ''.join([annotation.numpy().decode('utf-8') if isinstance(annotation, bytes) else str(annotation) for annotation in annotations])
        true_text= tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        # Calculate accuracy
        accuracy = calculate_accuracy(true_text, converted_prediction)

        # Calculate word error rate (WER)
        wer = calculate_wer(true_text, converted_prediction)

        # Display results
        st.info('Comparison with actual words:')
        st.text(f'Actual Words: {true_text}')
        st.text(f'Predicted Words: {converted_prediction}')
        st.text(f'Character-level accuracy: {accuracy}%')
        st.text(f'Word Error Rate (WER): {wer}%')

        
