import streamlit as st
import tensorflow as tf

def preprocess_image(data, shape = 224, scale = False):
    img = tf.io.read_file(data)
    img = tf.io.decode_image(img, channels = 3)
    img = tf.image.resize(img, [shape, shape])
    if scale:
        return img / 255.
    else:
        return img

st.title('👁️ Food-Vision 🍔')
st.header('Identify what\'s on your food image!')
st.write('To know more about this app, visit [**GitHub**](https://github.com/KamRoki/Food-Vision)')
image_data = st.file_uploader(label = 'Upload your food image.',
                        type = ['jpg', 'jpeg', 'png'])
#st.image(image_data)

img = preprocess_image(image_data)