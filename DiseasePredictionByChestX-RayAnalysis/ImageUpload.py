import streamlit as st
from PIL import Image
import numpy as np

from ImageSubmit import submit

#st.image("https://www.marnoa.ca/-/media/rj/rjl-advisor-sites/sites/m/a/marnoa/images/blogs/1-31-2024/danaher-thumbnail.jpg?h=238&w=800&la=en&hash=0C506AD8D9B97EF58ED2838CD0BA4BFB")
st.image("c:\\Users\\SMAJUMDAR\\GITREPO\\NextGen-IoT\\src\\UI-Framework\\FE\\Common\\img\\BCI Logo.png")
st.header("Analyze Chest X Ray Plate")

image=[]
response=""
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
        response = submit(image)
        print(response)

st.subheader("Expected Diasease : ") 
for i in response:
    st.markdown("- " + i)   

