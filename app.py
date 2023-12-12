import streamlit as st

st.title('👁️ Food-Vision 🍔')
st.header('Identify what\'s on your food image!')
st.write('To know more about this app, visit [**GitHub**](https://github.com/KamRoki/Food-Vision)')
file = st.file_uploader(label = 'Upload your food image.',
                        type = ['jpg', 'jpeg', 'png'])