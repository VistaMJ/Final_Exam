import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('OUR_model.h5')
    return model

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert('RGB')
    img = np.asarray(image)
    img = img / 255.0 
    img_reshape = np.reshape(img, (1, 128, 128, 3))
    prediction = model.predict(img_reshape)
    return prediction

model = load_model()

st.write("""
Detecting Tumor from head X-Ray Images
""")

file = st.file_uploader("Choose head x-ray photo from computer", type=["jpg", "png"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Normal', 'Tumor']
    string = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(string)
