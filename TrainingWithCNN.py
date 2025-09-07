
#---------------------------------------CNN----------------------------------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.utils import plot_model

# Directories for train and test datasets
train_dir = '/content/drive/MyDrive/Data/Train'
test_dir = '/content/drive/MyDrive/Data/Test'

# Image parameters
IMG_HEIGHT = 250
IMG_WIDTH = 250
BATCH_SIZE = 32

!rmdir '/content/drive/MyDrive/Data/.ipynb_checkpoints'
!rmdir '/content/drive/MyDrive/Data/Train/.ipynb_checkpoints'
!rmdir '/content/drive/MyDrive/Data/Test/.ipynb_checkpoints'

# ImageDataGenerator for training data with augmentation
train_image_generator = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ImageDataGenerator for test data (only rescaling)
test_image_generator = ImageDataGenerator(rescale=1.0/255)

# Flow from directory for training data
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Flow from directory for test data
test_data_gen = test_image_generator.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Define the CNN model # 32 - 64 - 128 - 128 -
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(train_data_gen.class_indices), activation='softmax')
])

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=250,
    validation_data=test_data_gen,
    validation_steps=test_data_gen.samples // BATCH_SIZE
)

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Visualize the model structure
plot_model(model, to_file='model_structure.png', rankdir="TB", show_shapes=True, show_layer_names=True)

# Display the model structure
from IPython.display import Image
Image(filename='model_structure.png')

# Evaluate the model on the test data
test_data_gen.reset()
predictions = model.predict(test_data_gen, steps=test_data_gen.samples // BATCH_SIZE + 1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data_gen.classes
class_labels = list(test_data_gen.class_indices.keys())

# Print classification report
report = classification_report(true_classes[:len(predicted_classes)], predicted_classes, target_names=class_labels)
print(report)

# Calculate test accuracy from classification report
accuracy = np.mean(predicted_classes == true_classes)
print(f'Test Accuracy: {accuracy:.4f}')

# Generate and plot confusion matrix
conf_matrix = confusion_matrix(true_classes[:len(predicted_classes)], predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the model
#model.save('/content/drive/MyDrive/my_trained_model.keras')
model.save('/content/drive/MyDrive/my_trained_model.h5')

