import tensorflow as tf
import numpy as np

from SpatialTransform import SpatialTransform


def main():
    x_train = np.load("datasets/x_train.npy")
    y_train = np.load("datasets/y_train.npy")

    x_test = np.load("datasets/x_test.npy")
    y_test = np.load("datasets/y_test.npy")

    # model = model_cnn()
    model = model_st()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    model.summary()

    checkpoint_filepath = "tmp/checkpoint"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor="val_accuracy", mode="max", save_best_only=True)

    model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)
    model.evaluate(x_test, y_test, verbose=2)

    # model.save("models/model_cnn")
    model.save("models/model_sp")


# model_cnn 5 epoch 64 batch
# eval: 88/88 - 9s - loss: 0.1135 - accuracy: 0.9606 - 9s/epoch - 107ms/step
def model_cnn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(50, 50, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(15, activation="softmax")
    ])
    return model


# model_st 15 epoch 64 batch
# eval: 88/88 - 21s - loss: 0.0809 - accuracy: 0.9885 - 21s/epoch - 236ms/step
def model_st():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((50, 50, 3), name="input"),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.Rescaling(1.0/255),

        SpatialTransform(128, (5, 5), 128, (5, 5), 200),
        tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="valid"),

        SpatialTransform(32, (5, 5), 64, (5, 5), 100),
        tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="valid"),

        SpatialTransform(32, (5, 5), 64, (5, 5), 100),
        tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2, 2), strides=2, padding="valid"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(15, activation="softmax", name="output")
    ])
    return model


if __name__ == "__main__":
    main()
