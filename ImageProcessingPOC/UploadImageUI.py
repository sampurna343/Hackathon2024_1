import streamlit as st

st.header("Analyze Chest X Ray Plate")

with st.sidebar:
    st.title("Images")
    file=st.file_uploader("Upload an inage to analyze",type="png")
