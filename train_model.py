import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 10


# ============================================================
# LOAD EMNIST BALANCED DATASET
# ============================================================

print("Loading EMNIST Balanced dataset...")

(ds_train, ds_test), ds_info = tfds.load(
    "emnist/balanced",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

print("Dataset loaded successfully.")


# ============================================================
# GET CLASS LABELS
# ============================================================

class_names = ds_info.features["label"].names

print("Number of classes:", len(class_names))
print("Classes:", class_names)


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess(image, label):

    # Convert image to float
    image = tf.cast(image, tf.float32)

    # Normalize pixels between 0 and 1
    image = image / 255.0

    # EMNIST images need orientation adjustment
    image = tf.transpose(image, perm=[1, 0, 2])

    return image, label


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

ds_train = ds_train.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

ds_test = ds_test.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)


ds_train = ds_train.shuffle(10000)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# ============================================================
# CREATE CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(28, 28, 1)
    ),

    # First convolution block
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution block
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution block
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    # 47 EMNIST Balanced classes
    tf.keras.layers.Dense(
        47,
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting CNN training...\n")

history = model.fit(
    ds_train,
    epochs=EPOCHS,
    validation_data=ds_test
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(ds_test)

print("\nTest Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "handwritten_character_model.keras"
)

print("\nModel saved successfully!")


# ============================================================
# SAVE CLASS LABELS
# ============================================================

with open("labels.json", "w") as file:

    json.dump(
        class_names,
        file
    )

print("Labels saved successfully!")

print("\nTraining completed successfully!")