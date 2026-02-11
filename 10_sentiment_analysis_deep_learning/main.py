import tensorflow as tf
from tensorflow import keras

# load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# pad sequences
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# build model
model = keras.Sequential([
    keras.layers.Embedding(10000, 32),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# train
model.fit(x_train, y_train, epochs=5, batch_size=128)

# evaluate
loss, acc = model.evaluate(x_test, y_test)

print("Accuracy:", acc)