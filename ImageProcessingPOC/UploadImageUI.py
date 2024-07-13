import streamlit as st
from PIL import Image
import numpy as np

st.header("Analyze Chest X Ray Plate")

with st.sidebar:
    st.title("Images")
    file=st.file_uploader("Upload an inage to analyze",type="png")
    if file is not None:
        image = Image.open(file)
        img_array = np.array(image)
        #image.show(file)
        st.image(image)


