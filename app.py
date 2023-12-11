import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

class_names = ['apple_pie',
                'baby_back_ribs',
                'baklava',
                'beef_carpaccio',
                'beef_tartare',
                'beet_salad',
                'beignets',
                'bibimbap',
                'bread_pudding',
                'breakfast_burrito',
                'bruschetta',
                'caesar_salad',
                'cannoli',
                'caprese_salad',
                'carrot_cake',
                'ceviche',
                'cheesecake',
                'cheese_plate',
                'chicken_curry',
                'chicken_quesadilla',
                'chicken_wings',
                'chocolate_cake',
                'chocolate_mousse',
                'churros',
                'clam_chowder',
                'club_sandwich',
                'crab_cakes',
                'creme_brulee',
                'croque_madame',
                'cup_cakes',
                'deviled_eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs_benedict',
                'escargots',
                'falafel',
                'filet_mignon',
                'fish_and_chips',
                'foie_gras',
                'french_fries',
                'french_onion_soup',
                'french_toast',
                'fried_calamari',
                'fried_rice',
                'frozen_yogurt',
                'garlic_bread',
                'gnocchi',
                'greek_salad',
                'grilled_cheese_sandwich',
                'grilled_salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot_and_sour_soup',
                'hot_dog',
                'huevos_rancheros',
                'hummus',
                'ice_cream',
                'lasagna',
                'lobster_bisque',
                'lobster_roll_sandwich',
                'macaroni_and_cheese',
                'macarons',
                'miso_soup',
                'mussels',
                'nachos',
                'omelette',
                'onion_rings',
                'oysters',
                'pad_thai',
                'paella',
                'pancakes',
                'panna_cotta',
                'peking_duck',
                'pho',
                'pizza',
                'pork_chop',
                'poutine',
                'prime_rib',
                'pulled_pork_sandwich',
                'ramen',
                'ravioli',
                'red_velvet_cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed_salad',
                'shrimp_and_grits',
                'spaghetti_bolognese',
                'spaghetti_carbonara',
                'spring_rolls',
                'steak',
                'strawberry_shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna_tartare',
                'waffles']

st.set_page_config(page_title = 'Food Vision',
                   page_icon = '👁️')

# SideBar
st.sidebar.title('What is Food Vision project?')
st.sidebar.write('''
It is an end-to-end **CNN Image Classification Model** notebook which identifies the food in your images.

The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, 
making a total of 101k images. The labels for the test images have been manually cleaned, while the 
training set contains some noise.

**Accuracy:** **`73.37%`**

**Model:** **`EfficientNetB1`**

**Dataset:** **`Food101`**
''')

st.sidebar.markdown('Created by **Kamil Stachurski**')

# Main Body
st.title('👁️ Food Vision 🍔')
st.header('Let\'s identify what\'s in your food photo!')
st.write('To know more about this app, visit [**GitHub**](https://github.com/KamRoki/Food-Vision)')

st.set_option('deprecation./showfileUploaderEncoding', False)

@st.cache(allow_output_mutation = True)

def load_model():
    model = tf.keras.models.load_model('models/fine_tuned_model.h5')
    return model

def predicting(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = tf.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction

model = load_model()
file = st.file_uploader('Upload your food image', type = ['jpg', 'jpeg', 'png'])

if file is None:
    st.text('Waiting for upload an image...')
else:
    slot = st.empty()
    slot.text('Running inference...')
    test_image = Image.open(file)
    st.image(test_image, caption = 'Input Image', width = 400)
    pred = predicting(np.asarray(test_image), model)
    pred_class = class_names[tf.argmax(pred)]
    pred_conf = tf.reduce_max(pred)
    output = 'Prediction: ' + pred_class + ' \nConfidence: ' + pred_conf
    slot.text('Done')
    st.success(output)