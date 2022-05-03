from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
# Create your views here.
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import pickle
import numpy as np
img_height, img_width=224,224

def home(request):
    return render(request,"index.html")


def predictImage(request):
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName) 
    testimage='.'+filePathName
    print("TEST IMAGE:"+testimage)
    img = image.load_img(testimage, target_size=(img_height, img_width))
    new_model = tf.keras.models.load_model('./models/final_model.h5')
    #MobileNetModelImagenet.h5
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
    return render(request,"index.html",{'filePathName':filePathName,'prediction':class_to_label[index]})

