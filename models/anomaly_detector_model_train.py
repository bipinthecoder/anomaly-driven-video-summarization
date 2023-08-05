import tensorflow as tf
from utils import helper_functions as hf
import Config as Config

# output shape and data type of the dataset
output_signature = (tf.TensorSpec(shape=(10, 256, 256, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(10, 256, 256, 1), dtype=tf.float32))

get_dataset_in_sequences = hf.get_dataset_in_sequences

# Converting the generator into a tf.data.Dataset
dataset = tf.data.Dataset.from_generator(
    get_dataset_in_sequences,
    output_signature=output_signature
)

batch_size = 4

# Repeat is used to make sure the generator doesn't run out of data while training
dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.experimental.AUTOTUNE)

# Loading the pre-trained LSTM Auto Encoder trained on USCD Pedestrian Dataset
pre_trained_model = tf.keras.models.load_model(Config.PRE_TRAINED_LSTM_AUTO_ENCODER_PATH)

print(pre_trained_model.summary())

'''Freezing Initial Layers and making following layers trainable '''

for layer in pre_trained_model.layers[:8]:
    layer.trainable = False

# Unfreezing the final the layers (TimeDistributed with Conv2DTranspose and Conv2D)
for layer in pre_trained_model.layers[8:]:
    layer.trainable = True

# Using the same compile function as used for the model while initial training
pre_trained_model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6),
                          metrics=['accuracy'])


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') <= 0.0014):
            print('\nLoss fell down 0.01, Stopping Training!')
            self.model.stop_training = True


# Initializing callback
callback = myCallback()

# Performing Training
history = pre_trained_model.fit(
    dataset,
    epochs=12,
    callbacks=[callback],
    steps_per_epoch=1650,
    shuffle=False
)

# Saving the model
pre_trained_model.save('./models/auto_encoder3.hdf5')