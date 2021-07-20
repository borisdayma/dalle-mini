#!/usr/bin/env python
# coding: utf-8

import random
from dalle_mini.backend import ServiceError, get_images_from_backend

import streamlit as st

# st.sidebar.title("DALL·E mini")

# sc = st.sidebar.beta_columns(2)
# st.sidebar.image('../img/logo.png', width=150)
# sc[1].write("  ")
# st.sidebar.markdown("Generate images from text")

st.sidebar.markdown("""
<style>
.aligncenter {
    text-align: center;
}
</style>
<p class="aligncenter">
    <img src="https://raw.githubusercontent.com/borisdayma/dalle-mini/main/img/logo.png"/>
</p>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
___
DALL·E mini is an AI model that generates images from any prompt you give!

<p style='text-align: center'>
Created by Boris Dayma et al. 2021
<a href="https://github.com/borisdayma/dalle-mini">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA">Project Report</a>
</p>
        """, unsafe_allow_html=True)

st.header('DALL·E mini')
st.subheader('Generate images from text')

prompt = st.text_input("What do you want to see?")

#TODO: I think there's an issue where we can't run twice the same inference (not due to caching) - may need to use st.form

DEBUG = False
if prompt != "" or (st.session_state.get("again", False) and prompt != ""):
    container = st.empty()
    container.markdown(f"Generating predictions for: **{prompt}**")

    try:
        backend_url = st.secrets["BACKEND_SERVER"]
        print(f"Getting selections: {prompt}")
        selected = get_images_from_backend(prompt, backend_url)

        cols = st.beta_columns(4)
        for i, img in enumerate(selected):
            cols[i%4].image(img)

        container.markdown(f"**{prompt}**")
        
        st.session_state["again"] = st.button('Again!', key='again_button')
    
    except ServiceError as error:
        container.text(f"Service unavailable, status: {error.status_code}")
    except KeyError:
        if DEBUG:
            container.markdown("""
            **Error: BACKEND_SERVER unset**

            Please, create a file called `.streamlit/secrets.toml` inside the app's folder and include a line to configure the server URL:
            ```
            BACKEND_SERVER="<server url>"
            ```
            """)
        else:
            container.markdown('Error -5, please try again or [report it](mailto:pcuenca-dalle@guenever.net).')
