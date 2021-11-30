import requests
from io import BytesIO
import base64
from PIL import Image

class ServiceError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code

def get_images_from_backend(prompt, backend_url):
    r = requests.post(
        backend_url, 
        json={"prompt": prompt}
    )
    if r.status_code == 200:
        images = r.json()["images"]
        images = [Image.open(BytesIO(base64.b64decode(img))) for img in images]
        return images
    else:
        raise ServiceError(r.status_code)
