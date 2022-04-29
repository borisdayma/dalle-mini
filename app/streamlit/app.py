#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from backend import ServiceError, get_images_from_backend

st.sidebar.markdown(
    """
<style>
.aligncenter {
    text-align: center;
}
</style>
<p class="aligncenter">
    <img src="https://raw.githubusercontent.com/borisdayma/dalle-mini/main/img/logo.png"/>
</p>
""",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """
___
<p style='text-align: center'>
DALL·E mini is an AI model that generates images from any prompt you give!
</p>

<p style='text-align: center'>
Created by Boris Dayma et al. 2021-2022
<br/>
<a href="https://github.com/borisdayma/dalle-mini" target="_blank">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA" target="_blank">Project Report</a>
</p>
        """,
    unsafe_allow_html=True,
)

st.header("DALL·E mini")
st.subheader("Generate images from text")

prompt = st.text_input("What do you want to see?")

DEBUG = False
if prompt != "":
    container = st.empty()
    container.markdown(
        f"""
        <style> p {{ margin:0 }} div {{ margin:0 }} </style>
        <div data-stale="false" class="element-container css-1e5imcs e1tzin5v1">
        <div class="stAlert">
        <div role="alert" data-baseweb="notification" class="st-ae st-af st-ag st-ah st-ai st-aj st-ak st-g3 st-am st-b8 st-ao st-ap st-aq st-ar st-as st-at st-au st-av st-aw st-ax st-ay st-az st-b9 st-b1 st-b2 st-b3 st-b4 st-b5 st-b6">
        <div class="st-b7">
        <div class="css-whx05o e13vu3m50">
        <div data-testid="stMarkdownContainer" class="css-1ekf893 e16nr0p30">
                <img src="https://raw.githubusercontent.com/borisdayma/dalle-mini/main/app/streamlit/img/loading.gif" width="30"/>
                Generating predictions for: <b>{prompt}</b>
        </div>
        </div>
        </div>
        </div>
        </div>
        </div>
        <small><i>Predictions may take up to 5mn under high load. Please stand by.</i></small>
    """,
        unsafe_allow_html=True,
    )

    try:
        backend_url = st.secrets["BACKEND_SERVER"] + "/generate"
        response = get_images_from_backend(prompt, backend_url)
        selected = response["images"]
        version = response["version"]

        margin = 0.1  # for better position of zoom in arrow
        n_columns = 3
        cols = st.columns([1] + [margin, 1] * (n_columns - 1))
        for i, img in enumerate(selected):
            cols[(i % n_columns) * 2].image(img)
        container.markdown(f"**{prompt}**")

        # st.sidebar.markdown(
        #    f"<small><center>{version}</center></small>", unsafe_allow_html=True
        # )

        # st.markdown(
        #    f"""
        # These results have been obtained using model `{version}` from [an ongoing training run](https://wandb.ai/dalle-mini/dalle-mini/runs/mheh9e55).
        # """
        # )

        st.button("Again!", key="again_button")

    except ServiceError as error:
        container.text(f"Service unavailable, status: {error.status_code}")
    except KeyError:
        if DEBUG:
            container.markdown(
                """
            **Error: BACKEND_SERVER unset**

            Please, create a file called `.streamlit/secrets.toml` inside the app's folder and include a line to configure the server URL:
            ```
            BACKEND_SERVER="<server url>"
            ```
            """
            )
        else:
            container.markdown(
                "Error -5, please try again or [report it](mailto:pcuenca-dalle@guenever.net)."
            )
