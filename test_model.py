import Config as Config
import tensorflow as tf

model_path = Config.FINE_TUNED_LSTM_AUTO_ENCODER_PATH

# Loading the model from keras
model = tf.keras.models.load_model(model_path)


