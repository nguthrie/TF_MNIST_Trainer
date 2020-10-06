import tensorflow as tf
import numpy as np


def mnist_digits():

    # TODO Find a new basic example that trains quickly using a Covnet

    # load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # choose model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile the model
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # train the model
    model.fit(x_train, y_train, epochs=3)

    # how well is our model performing
    model.evaluate(x_test, y_test, verbose=2)

    # testing the model
    img = x_test[1]

    print("True label:", y_test[1])

    img = (np.expand_dims(img, 0))

    # set probability model for later predictions
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions_single = probability_model.predict(img)

    print("Probability vector:", predictions_single[0].round(4))
    print("Prediction from model:", predictions_single.argmax())


if __name__ == "__main__":

    mnist_digits()

