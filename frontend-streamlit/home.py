import requests
import streamlit as st
import os

from PIL import Image
from decouple import config
from app.services.pushtext import Web_ruted

# set the api endpoint
if "API_ENDPOINT" not in os.environ:
    os.environ["API_ENDPOINT"] = config("API_ENDPOINT")

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)

st.write("# Webs classification Tool! 👋")
st.text("This app is used to classification webs")

col1, col2 = st.columns(2)

with col1:

    text_URL = st.text_input(
        "Enter the URL 👇",
        label_visibility="visible",
        disabled=False,
        placeholder="Webside to classification",
    )
    button = st.button("Push URL")
    if button and text_URL:
        response, response_json= Web_ruted.pushpost(url=text_URL)
        st.write(response_json)

        is_success = response.status_code == 200
        if is_success:
            st.success("Report Web!", icon="✅")  # display a success message
            st.balloons()
        else:
            st.error("Something went wrong!", icon="❌")  # display an error message
            st.write(response.json()["detail"])

with col2:
    image = Image.open('image.png')
    st.image(image, caption='Classification Histogram')
