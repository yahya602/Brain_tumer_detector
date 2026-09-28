import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Brain Tumer Detector")

model = YOLO("best.pt")

uploaded = st.file_uploader("Uplaod Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Submit"):
        with st.spinner("Detecting..."):
            results = model.predict(img, conf=0.25)
        st.image(results[0].plot()[..., ::-1], caption="Result")