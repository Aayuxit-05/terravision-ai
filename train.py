import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================
# SETTINGS
# ==============================

DATASET_DIR = "EuroSAT_RGB"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# ==============================
# LOAD DATASET
# ==============================

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("\nClasses:")
print(class_names)

# ==============================
# PERFORMANCE
# ==============================

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
validation_data = validation_data.prefetch(buffer_size=AUTOTUNE)

# ==============================
# DATA AUGMENTATION
# ==============================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==============================
# TRANSFER LEARNING
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = keras.Sequential([
    data_augmentation,

    layers.Rescaling(
        1.0 / 127.5,
        offset=-1
    ),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.2),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# TRAIN
# ==============================

print("\nStarting training...\n")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=3
)

# ==============================
# SAVE MODEL
# ==============================

model.save("terravision_model.keras")

print("\nTraining complete!")
print("Model saved as terravision_model.keras")