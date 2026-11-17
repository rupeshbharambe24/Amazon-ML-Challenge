from io import BytesIO
from PIL import Image
import requests

_SESSION = requests.Session()

def fetch_image_or_gray(url: str, size: int = 224):
    try:
        r = _SESSION.get(url, timeout=6)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), color=(128, 128, 128))
    return img
