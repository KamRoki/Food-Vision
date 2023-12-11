import tensorflow as tf
import streamlit as st

def load_and_prep_image(image, img_shape = 224, scale = False):
    # Decode it into a tensor
    img = tf.image.decode_image(img, channels = 3)
    # Resize the image
    img = tf.image.resize(img, [img_size, img_size])
    if scale:
        return img / 255.
    else:
        return img

@st.cache(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep_image(image, img_shape = 224, scale = False)
    image = tf.cast(tf.expand_dims(image, axis = 0), tf.float32)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    return pred_class, pred_conf

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

# Main Body
st.title('👁️ Food Vision 🍔')
st.header('Let\'s identify what\'s in your food photo!')
st.write('To know more about this app, visit [**GitHub**](https://github.com/KamRoki/Food-Vision)')

file = st.file_uploader(label = 'Upload your food image.',
                        type = ['jpg', 'jpeg', 'png'])

model = tf.keras.models.load_model('./models/fine_tuned_model.hdf5')

st.sidebar.markdown('Created by **Kamil Stachurski**')

if not file:
    st.warning('Please upload an image...')
else:
    image = file.read()
    st.image(image, use_column_width = True)
    pred_button = st.button('Predict food class')

if pred_button:
    pred_class, pred_conf = predicting(image, model)
    st.success(f'Prediction: {pred_class} \nConfidence: {pred_conf * 100:.2f}%')