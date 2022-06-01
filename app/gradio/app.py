#!/usr/bin/env python
# coding: utf-8

import gradio as gr

block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

with block:
    gr.Markdown("<h1><center>DALL·E mini</center></h1>")
    gr.Markdown(
        "DALL·E mini is an AI model that generates images from any prompt you give!"
    )
    with gr.Group():
        with gr.Box():
            with gr.Row().style(mobile_collapse=False, equal_height=True):

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                ).style(
                    container=False,
                )
                btn = gr.Button("Run", variant="primary")
        gallery = gr.Gallery(label="Generated images", show_label=False).style(
            grid=[3], height="auto"
        )
        btn.click(
            None,
            _js="""
        async (text) => {
            response = await fetch('http://backend.dallemini.ai/generate', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: text
                })
            });
            response = await response.json()
            let imgs = response.images.map(r => "data:image/png;base64," + r)
            return imgs
        }
        """,
            inputs=text,
            outputs=gallery,
        )

    gr.Markdown(
        """___
   <p style='text-align: center'>
   Created by <a href="https://twitter.com/borisdayma" target="_blank">Boris Dayma</a> et al. 2021-2022
   <br/>
   <a href="https://github.com/borisdayma/dalle-mini" target="_blank">GitHub</a> | <a href="https://wandb.ai/dalle-mini/dalle-mini/reports/DALL-E-mini-Generate-images-from-any-text-prompt--VmlldzoyMDE4NDAy" target="_blank">Project Report</a>
   </p>"""
    )


import json

blocks_config = block.get_config_file()
blocks_config["dev_mode"] = False
blocks_config = json.dumps(blocks_config)
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="utf-8" />
		<meta
			name="viewport"
			content="width=device-width, initial-scale=1, shrink-to-fit=no, maximum-scale=1"
		/>

		
		<script>
			window.__gradio_mode__ = "app";
            window.gradio_config = {blocks_config};
        </script>

		<link rel="preconnect" href="https://fonts.googleapis.com" />
		<link
			rel="preconnect"
			href="https://fonts.gstatic.com"
			crossorigin="anonymous"
		/>
		<link
			href="https://fonts.googleapis.com/css?family=Source Sans Pro"
			rel="stylesheet"
		/>
		<link
			href="https://fonts.googleapis.com/css?family=IBM Plex Mono"
			rel="stylesheet"
		/>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js"></script>
		<script type="module" crossorigin src="https://gradio.s3-us-west-2.amazonaws.com/3.0.9b9/assets/index.01245d42.js"></script>
		<link rel="stylesheet" href="https://gradio.s3-us-west-2.amazonaws.com/3.0.9b9/assets/index.cbea297d.css">
        <style>
            footer img {{
                display: none !important;
            }}
        </style>
	</head>

	<body
		style="
			margin: 0;
			padding: 0;
			display: flex;
			flex-direction: column;
			flex-grow: 1;
		"
	>
		<div
			id="root"
			style="display: flex; flex-direction: column; flex-grow: 1"
		></div>
	</body>
</html>
"""

with open("index.html", "w") as index_html:
    index_html.write(HTML_TEMPLATE)
