import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

#  Dataset Path (Each class in a separate Folder)
base_dir = "/content/drive/MyDrive/Data2"

# پارامترها
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16     # Less for not very large dataset
EPOCHS = 40         # Not that We repeat the whole process 5 times!
N_SPLITS = 5
SEED = 42

# Creating lists from all images and labels
filepaths = []
labels = []
for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_name)

data_df = pd.DataFrame({"filename": filepaths, "class": labels})

print(f"Total images: {len(data_df)}")
print(data_df['class'].value_counts())

# DataGenerators
train_image_generator = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_image_generator = ImageDataGenerator(rescale=1.0/255)

# Lists for storing results
acc_scores = []
f1_scores = []

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, test_idx) in enumerate(kf.split(data_df)):
    print(f"\n===== Fold {fold+1} / {N_SPLITS} =====")

    train_df = data_df.iloc[train_idx]
    test_df = data_df.iloc[test_idx]

    # Generator for Train
    train_gen = train_image_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    # Generator for Test
    test_gen = test_image_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col="filename",
        y_col="class",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Constructing MobileNetV2
    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                             include_top=False,
                             weights="imagenet")
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(len(train_gen.class_indices), activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Train
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        verbose=1
    )

    # Prediction on Test data
    preds = np.argmax(model.predict(test_gen), axis=1)
    true = test_gen.classes

    acc = np.mean(preds == true)
    report = classification_report(true, preds, output_dict=True, zero_division=0)

    acc_scores.append(acc)
    f1_scores.append(report["macro avg"]["f1-score"])

    print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    print(f"Fold {fold+1} Macro-F1: {report['macro avg']['f1-score']:.4f}")

# Final results
print("\n===== Final Results (5-Fold CV) =====")
print(f"Mean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"Mean Macro-F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")