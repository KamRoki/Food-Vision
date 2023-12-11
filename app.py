import streamlit as st
import tensorflow as tf
from PIL import Image

def load_image():
    uploaded_file = st.file_uploader(label = 'Choose your food image')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def load_model():
    model = tf.keras.models.load_model('models/fine_tuned_model.h5')
    return model

def preprocess(image):
    image = tf.image.resize(image, size = [224, 224])
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis = 0)
    return image



def main():
    st.title('👁️ Food Vision 🍔')
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
    model = load_model()
    img = load_image()
    input_tensor = preprocess(img)
    pred_probs = model.predict(input_tensor)
    pred_class = class_names[tf.argmax(pred_probs)]
    pred_conf = tf.reduce_max(pred)
    result = st.button('Predict')





if __name__ == '__main__':
    main()