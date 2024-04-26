import tensorflow as tf


# data preparation
train_path = 'data/train'
test_path = 'data/validation'

IMG_SIZE = (48, 48)
NUM_CLASS = 7
BATCH_SIZE = 32

# creating training and validation loader
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    # color_mode = "grayscale",
    seed=99,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    # color_mode = "grayscale",
    seed=99,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_path,
    # color_mode = "grayscale",
    seed=99,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE)

# prefetching and cache the data for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
# def preprocess(x, y):
#     x = tf.image.rgb_to_grayscale(x)
#     x = x/255.0
#     return (x, y)

# train_dataset = train_dataset.map(preprocess)
# validation_dataset = validation_dataset.map(preprocess)
# test_dataset = test_dataset.map(preprocess)

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# creating cnn model
model = tf.keras.Sequential([
    tf.keras.layers.Input([IMG_SIZE[0], IMG_SIZE[1], 3]),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASS, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# training the model
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# evaluating
_, accuracy = model.evaluate(test_dataset)
print(f"The test accuracy is {round(accuracy*100, 2)}%")


import os
import shutil

dir_path = "saved_model"
if not os.path.exists(dir_path):
    # If the directory does not exist, create it
    os.makedirs(dir_path)

# removing the previously stored models
tf_model_path = "saved_model/tf_model.h5"
if os.path.exists(tf_model_path):
    shutil.rmtree("saved_model")

# saving the model
model.save('saved_model/tf_model.h5')
