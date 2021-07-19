#!/usr/bin/env python
# coding: utf-8

import random
from dalle_mini.backend import ServiceError, get_images_from_backend
from dalle_mini.helpers import captioned_strip

import streamlit as st

# Controls

# num_images = st.sidebar.slider("Candidates to generate", 1, 64, 8, 1)
# num_preds = st.sidebar.slider("Best predictions to show", 1, 8, 1, 1)

st.sidebar.markdown('Visit [our report](https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA)')

prompt = st.text_input("What do you want to see?")

if prompt != "":
    st.write(f"Generating candidates for: {prompt}")

    try:
        backend_url = st.secrets["BACKEND_SERVER"]
        print(f"Getting selections: {prompt}")
        selected = get_images_from_backend(prompt, backend_url)
        preds = captioned_strip(selected, prompt)
        st.image(preds)
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
