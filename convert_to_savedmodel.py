import tensorflow as tf

model = tf.keras.models.load_model("artifacts/cats_and_dogs.h5")

tf.saved_model.save(model, "cats_and_dogs")