import streamlit as st
from PIL import Image
import numpy as np

from ImageSubmit import submit

st.header("Analyze Chest X Ray Plate")

image=[]
with st.sidebar:
    st.title("Image Submit")
    file=st.file_uploader("Upload an inage to analyze",type=["png","jpg","jpeg"])
    if file is not None:
        image = Image.open(file)
        img_array = np.array(image)
        #image.show()
        st.image(image)
    submitButton=st.button(label="Submit", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False)
    if submitButton:
        submit(image)

