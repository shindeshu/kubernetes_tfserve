import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import grpc

from protobuf_utils import np_to_protobuf
import requests
from PIL import Image
from io import BytesIO
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from fastapi import FastAPI
import uvicorn
app = FastAPI()


host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

from pydantic import BaseModel

class Item(BaseModel):
    url: str

def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'cats_and_dogs'
    pb_request.model_spec.signature_name = 'serving_default'

    print(X.shape, X[0].shape)
    proto_input = np_to_protobuf(X[0])

    pb_request.inputs['input_1'].CopyFrom(proto_input)
    return pb_request

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

def postprocess(preds: list):
    classes = ['cat', 'dog']
    return dict(zip(classes, preds))

def prepare_response(pb_response):
    classes = ['cat', 'dog']
    preds = pb_response.outputs['dense_3'].float_val
    return dict(zip(classes, preds))


def predict(url):
    X = load_image(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


@app.post('/predict/')
def predict_endpoint(item: Item):
    url = item.url
    result = predict(url)
    return result


if __name__ == '__main__':
    # url = "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg"
    # response = predict(url)
    # print(response)
    uvicorn.run(app, host='0.0.0.0', port=9696)