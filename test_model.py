import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# LOAD MODEL
# ==============================

model = tf.keras.models.load_model("terravision_model.keras")

# ==============================
# SETTINGS
# ==============================

DATASET_DIR = "EuroSAT_RGB"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# ==============================
# LOAD VALIDATION DATA
# ==============================

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = validation_data.class_names

# ==============================
# PREDICT IN ONE PASS
# ==============================

true_labels = []
predicted_labels = []

print("\nEvaluating model...\n")

for images, labels in validation_data:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    )
)

# ==============================
# CONFUSION MATRIX
# ==============================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

print(
    confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(len(class_names))
    )
)