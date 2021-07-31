#!/usr/bin/env python
# coding: utf-8

import random
from dalle_mini.backend import ServiceError, get_images_from_backend

import streamlit as st

# streamlit.session_state is not available in Huggingface spaces.
# Session state hack https://huggingface.slack.com/archives/C025LJDP962/p1626527367443200?thread_ts=1626525999.440500&cid=C025LJDP962

from streamlit.report_thread import get_report_ctx
def query_cache(q_emb=None):
    ctx = get_report_ctx()
    session_id = ctx.session_id
    session = st.server.server.Server.get_current()._get_session_info(session_id).session
    if not hasattr(session, "_query_state"):
        setattr(session, "_query_state", q_emb)
    if q_emb:
        session._query_state = q_emb
    return session._query_state

def set_run_again(state):
    query_cache(state)

def should_run_again():
    state = query_cache()
    return state if state is not None else False

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
<p style='text-align: center'>
DALL·E mini is an AI model that generates images from any prompt you give!
</p>

<p style='text-align: center'>
Created by Boris Dayma et al. 2021
<br/>
<a href="https://github.com/borisdayma/dalle-mini" target="_blank">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini--Vmlldzo4NjIxODA" target="_blank">Project Report</a>
</p>
        """, unsafe_allow_html=True)

st.header('DALL·E mini')
st.subheader('Generate images from text')

prompt = st.text_input("What do you want to see?")

#TODO: I think there's an issue where we can't run twice the same inference (not due to caching) - may need to use st.form

DEBUG = False
if prompt != "" or (should_run_again and prompt != ""):
    container = st.empty()
    container.markdown(f"""
        <style> p {{ margin:0 }} </style>
        <p style="background-color:#FCF3CF;">
        <img src="https://raw.githubusercontent.com/borisdayma/dalle-mini/main/app/img/loading.gif" width="30"/>
        Generating predictions for: <b>{prompt}</b>
        </p>
        <small><i>Predictions may take up to 40s under high load. Please, stand by.</i></small>
    """, unsafe_allow_html=True)
    # container.markdown("more markdown")

    # container.markdown(f"Generating predictions for: **{prompt}**")
    # container.info(f"this is an info field")
    # container.warning(f"this is a warning field")

    try:
        backend_url = st.secrets["BACKEND_SERVER"]
        print(f"Getting selections: {prompt}")
        selected = get_images_from_backend(prompt, backend_url)

        cols = st.beta_columns(4)
        for i, img in enumerate(selected):
            cols[i%4].image(img)

        container.markdown(f"**{prompt}**")
        
        set_run_again(st.button('Again!', key='again_button'))
    
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
