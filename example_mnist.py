import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
import numpy as np

from clever import clever_t, clever_u

model = keras.Sequential([
    layers.Flatten(input_shape=[28,28,1]),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.categorical_crossentropy,
    metrics=["accuracy"]
)
model.build()



(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

rows, cols = 28, 28
num_classes = 10
batch_size=128

if keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, rows, cols)
    x_test = x_test.reshape(x_test.shape[0], 1, rows, cols)
    input_shape = (1, rows, cols)
else:
    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
    input_shape = (rows, cols, 1)

x_train = np.float32(x_train / 255.0)
x_test = np.float32(x_test / 255.0)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)



x_0, y_0 = np.reshape(x_train[0], [1,28,28,1]), np.argmax(y_train[0])

targeted_clever_score = clever_t(
    model,
    example = x_0,
    predicted_label = y_0,
    target_label = (y_0 + 1 % 10),
    batch_size = 32,
    samples_per_batch = 100,
    perturbation_norm = 2,
    maximum_perturbation = 5,
)
print(targeted_clever_score)
untargeted_clever_score = clever_u(
    model,
    example = x_0,
    predicted_label = y_0,
    num_labels = 10,
    batch_size = 32,
    samples_per_batch = 100,
    perturbation_norm = 2,
    maximum_perturbation = 5,
)
print(untargeted_clever_score)