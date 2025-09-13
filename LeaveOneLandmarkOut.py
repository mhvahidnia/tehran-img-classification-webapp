# ------------------- MobileNetV2 with Leave-One-Landmark-Out + OOD metrics ----------------
import os
import shutil
import numpy as np
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ----------------- USER PARAMETERS -----------------
BASE_DIR = '/content/drive/MyDrive/Data/Train'   
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 40   
VAL_SPLIT = 0.15
UNKNOWN_THRESHOLD = 0.5   
SAVE_PLOTS_DIR = '/content/ood_plots'
os.makedirs(SAVE_PLOTS_DIR, exist_ok=True)
# ---------------------------------------------------

all_items = sorted(os.listdir(BASE_DIR))
all_classes = [d for d in all_items if os.path.isdir(os.path.join(BASE_DIR, d))]
print("Detected classes:", all_classes)

# Generators
train_image_generator = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VAL_SPLIT
)

test_image_generator = ImageDataGenerator(rescale=1.0/255)

summary = []

def link_or_copy(src, dst):
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copytree(src, dst)

for left_out in all_classes:
    print("\n\n===== Leaving out class:", left_out, "=====")

    train_temp = '/content/train_temp'
    test_temp = '/content/test_temp'
    if os.path.exists(train_temp):
        shutil.rmtree(train_temp)
    if os.path.exists(test_temp):
        shutil.rmtree(test_temp)
    os.makedirs(train_temp, exist_ok=True)
    os.makedirs(test_temp, exist_ok=True)

    for c in all_classes:
        src = os.path.join(BASE_DIR, c)
        if c == left_out:
            dst = os.path.join(test_temp, c)
            link_or_copy(src, dst)
        else:
            dst = os.path.join(train_temp, c)
            link_or_copy(src, dst)

    train_gen = train_image_generator.flow_from_directory(
        train_temp,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',   
        shuffle=True
    )
    val_gen = train_image_generator.flow_from_directory(
        train_temp,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    test_gen = test_image_generator.flow_from_directory(
        test_temp,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"Found {train_gen.samples} images for training (across {len(train_gen.class_indices)} classes).")
    print(f"Found {val_gen.samples} images for validation.")
    print(f"Found {test_gen.samples} images for testing (left-out class).")

    base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(len(train_gen.class_indices), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1
    )

    preds_val = model.predict(val_gen, verbose=0)
    preds_test = model.predict(test_gen, verbose=0)

    maxprob_val = np.max(preds_val, axis=1) if preds_val.size else np.array([])
    maxprob_test = np.max(preds_test, axis=1) if preds_test.size else np.array([])

    predicted_classes = np.argmax(preds_test, axis=1) if preds_test.size else np.array([])
    true_classes = test_gen.classes if hasattr(test_gen, 'classes') else np.zeros(len(predicted_classes), dtype=int)
    acc = float(np.mean(predicted_classes == true_classes)) if predicted_classes.size else 0.0

    mean_known = float(np.mean(maxprob_val)) if maxprob_val.size else float('nan')
    mean_unknown = float(np.mean(maxprob_test)) if maxprob_test.size else float('nan')
    frac_unknown_below_thresh = float(np.mean(maxprob_test < UNKNOWN_THRESHOLD)) if maxprob_test.size else float('nan')


    auroc = None
    try:
        if maxprob_val.size and maxprob_test.size:
            y = np.concatenate([np.ones_like(maxprob_val), np.zeros_like(maxprob_test)])  # 1=known, 0=unknown
            scores = np.concatenate([maxprob_val, maxprob_test])
            # roc_auc requires both classes present
            auroc = float(roc_auc_score(y, scores))
    except Exception as e:
        auroc = None

    print(f"Simple accuracy on left-out class ({left_out}): {acc:.4f}")
    print(f"mean max-prob (known val): {mean_known:.4f}")
    print(f"mean max-prob (left-out / unknown): {mean_unknown:.4f}")
    print(f"fraction unknown with max_prob < {UNKNOWN_THRESHOLD}: {frac_unknown_below_thresh:.3f}")
    print(f"AUROC (known vs unknown by max_prob): {auroc}")

    plt.figure(figsize=(8,4))
    if maxprob_val.size:
        plt.hist(maxprob_val, bins=30, alpha=0.6, label='known (val)')
    if maxprob_test.size:
        plt.hist(maxprob_test, bins=30, alpha=0.6, label=f'left-out: {left_out}')
    plt.xlabel('max softmax probability')
    plt.ylabel('count')
    plt.title(f'Confidence distribution - left-out: {left_out}')
    plt.legend()
    plot_path = os.path.join(SAVE_PLOTS_DIR, f'confidence_{left_out}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence histogram to: {plot_path}")

    summary.append({
        'left_out': left_out,
        'acc_on_leftout': acc,
        'mean_known_maxprob': mean_known,
        'mean_unknown_maxprob': mean_unknown,
        'frac_unknown_below_thresh': frac_unknown_below_thresh,
        'auroc': auroc,
        'n_train_samples': train_gen.samples,
        'n_val_samples': val_gen.samples,
        'n_test_samples': test_gen.samples
    })

    try:
        shutil.rmtree(train_temp)
        shutil.rmtree(test_temp)
    except Exception:
        pass

print("\n\n===== Summary (Leave-One-Landmark-Out + OOD) =====")
for r in summary:
    print(f"{r['left_out']}: acc_leftout={r['acc_on_leftout']:.4f}, mean_known={r['mean_known_maxprob']:.3f}, "
          f"mean_unknown={r['mean_unknown_maxprob']:.3f}, frac_unknown<{UNKNOWN_THRESHOLD}={r['frac_unknown_below_thresh']:.3f}, auroc={r['auroc']}")

import csv
out_csv = '/content/leave_one_landmark_summary.csv'
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)
print("Saved CSV summary to:", out_csv)
