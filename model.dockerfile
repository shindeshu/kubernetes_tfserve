FROM tensorflow/serving:2.7.0

COPY cats_and_dogs /models/cats_and_dogs/1

ENV MODEL_NAME="cats_and_dogs"