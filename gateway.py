import os

import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
app = FastAPI()

host = os.getenv('TF_SERVING_HOST', 'localhost:8501')
classes = ['cat', 'dog']

class Item(BaseModel):
    url: str = "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"


def load_image(path, from_url=True, process=True):
    """
    Custom preprocessing function. 
    """
    if from_url:
        response = requests.get(path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(path)
    if process:
        img = img.resize((224,224), Image.Resampling.LANCZOS)
        img = np.array(img, dtype=np.float32)
        img = img * (1./255)
    return np.asarray([img])


def predict(url):
    X = load_image(url)
    model_url = f"http://{host}/v1/models/cats_and_dogs:predict"
    response = requests.post(model_url, data=json.dumps({"instances": X.tolist()}))
    predictions = response.json()['predictions'][0]
    return dict(zip(classes, predictions))


@app.post('/predict/')
def predict_endpoint(item: Item):
    url = item.url
    result = predict(url)
    return result


if __name__ == '__main__':
    url = "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"
    response = predict(url)
    print(response)
    # uvicorn.run(app, host='0.0.0.0', port=9696)