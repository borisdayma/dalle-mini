#!/usr/bin/env python
# coding: utf-8

import random
from dalle_mini.backend import ServiceError, get_images_from_backend
from dalle_mini.helpers import captioned_strip

import streamlit as st


st.sidebar.markdown('Visit [our report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA)')

st.header('DALL-E mini')
st.subheader('Generate images from a text prompt')

prompt = st.text_input("What do you want to see?")

#TODO: I think there's an issue where we can't run twice the same inference (not due to caching) - may need to use st.form

if prompt != "":
    st.write(f"Generating candidates for: {prompt}")

    try:
        backend_url = st.secrets["BACKEND_SERVER"]
        print(f"Getting selections: {prompt}")
        selected = get_images_from_backend(prompt, backend_url)

        cols = st.beta_columns(4)
        for i, img in enumerate(selected):
            cols[i%4].image(img)
        
    except ServiceError as error:
        st.write(f"Service unavailable, status: {error.status_code}")
    except KeyError:
        st.write("""
        **Error: BACKEND_SERVER unset**

        Please, create a file called `.streamlit/secrets.toml` inside the app's folder and include a line to configure the server URL:
        ```
        BACKEND_SERVER="<server url>"
        ```
        """)
