import streamlit as st
from tensorflow import keras
from PIL import Image, ImageFilter
from utils import image_prep
import numpy as np
import string
import os

dirname = os.path.dirname(__file__)
modelpath = os.path.join(dirname, 'models')


st.title("Deteksi ASL Alphabet")

st.subheader('Image Recognition by Nadhiar')

# @st.cache
model = keras.models.load_model(modelpath)

st.write("")
st.write('Berbasis CNN model')

label_dic = {i:string.ascii_uppercase[i] for i in range(26)}
label_dic.pop(9)
label_dic.pop(25)

uploaded_file = st.file_uploader("Pilih Gambar (.jpg)..." , type = 'jpg')

if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Gambar yang dipilih', use_column_width=True)
    st.write("")
    st.write("Convert gambar ke grayscale 28x28 pixel image . . . . . . ")
    prepped_img = image_prep.imageprepare(uploaded_image)
    st.write("Klasifikasi Gambar . . . . .")
    prediction = np.argmax(model.predict(prepped_img))
    alphabet = label_dic[prediction]
    st.subheader(f'Gambar tersebut adalah alphabet  {alphabet}')
    



