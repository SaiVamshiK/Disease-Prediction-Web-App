import streamlit as st
from keras.models import load_model
from keras.preprocessing import image as dif
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import pickle
import os
from django.core.files.storage import FileSystemStorage
import numpy as np
from PIL import Image as kmp
from keras.preprocessing.image import load_img, img_to_array
img_height, img_width=224,224


def predictImage(name):
    testimage = "media/"+name
    img = dif.load_img(testimage, target_size=(img_height, img_width))
    new_model = tf.keras.models.load_model('./models/final_model.h5')
    img = img_to_array(img)
    samples_to_predict = []
    samples_to_predict.append(img)
    samples_to_predict = np.array(samples_to_predict)
    print(samples_to_predict.shape)
    predictions = new_model.predict(samples_to_predict)
    class_to_label = {
        0:'MildDemented',
        1:'ModerateDemented',
        2:'NonDemented',
        3:'VeryMildDemented'
    }
    index = 0
    maxSoFar = predictions[0][0]
    for i in range(1,4):
        if maxSoFar < predictions[0][i]:
            maxSoFar = predictions[0][i]
            index = i
    print("prediction : ", class_to_label[index])
    return class_to_label[index]


st.title("Alzheimer's disease detection")
st.text("Upload a brain MRI Image for image classification")

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = kmp.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    with open(os.path.join("media",uploaded_file.name),"wb") as f: 
        f.write(uploaded_file.getbuffer())
    label = predictImage(uploaded_file.name)
    st.write('The given brain MRI image is : ',label)