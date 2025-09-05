import os
import shutil
from sklearn.model_selection import train_test_split

# Define directories
base_dir = '/content/drive/MyDrive/Data'
train_dir = '/content/drive/MyDrive/Data/Train'
test_dir = '/content/drive/MyDrive/Data/Test'

# Create Train and Test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Split the data
for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if os.path.isdir(class_dir):
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        for image in train_images:
            shutil.move(os.path.join(class_dir, image), os.path.join(train_class_dir, image))

        for image in test_images:
            shutil.move(os.path.join(class_dir, image), os.path.join(test_class_dir, image))