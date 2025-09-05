#-------------------MobileNetV2 ----------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Directories for train and test datasets
train_dir = '/content/drive/MyDrive/Data/Train'
test_dir = '/content/drive/MyDrive/Data/Test'

# Image parameters - Make sure these are correctly set for MobileNetV2
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16  #32

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
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Ensure these match MobileNetV2 input
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Flow from directory for test data
test_data_gen = test_image_generator.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Ensure these match MobileNetV2 input
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Load MobileNetV2 model pre-trained on ImageNet
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers for our classification task
model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),  # Added InputLayer to define input shape
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(len(train_data_gen.class_indices), activation='softmax')
])

# Print the model summary
model.summary()

# Visualize the model structure
tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# Display the model structure
from IPython.display import Image
Image(filename='model_structure.png')

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=100,  # You can increase the number of epochs if needed
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

# Evaluate the model on the test data
test_data_gen.reset()
predictions = model.predict(test_data_gen, steps=test_data_gen.samples // BATCH_SIZE + 1)
predicted_classes = np.argmax(predictions, axis=1)
# Use test_data_gen.classes to get the true labels
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
model.save('/content/drive/MyDrive/my_trained_mobilenet_model.h5')