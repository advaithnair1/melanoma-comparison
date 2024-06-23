import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog


# dataset i used: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
path_to_benign = "benign images here"
path_to_malignant = "malignant images here"

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((300, 300))  # Resize to 300x300
            img_array = np.array(img)  # Convert to numpy array
            hog_features = hog(img_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)  # Extract HOG features
            images.append(hog_features)
            label = 1 if 'malignant' in folder else 0  # Label: 1 for malignant, 0 for benign
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

def augment_image(image):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    image = image.reshape((1, ) + image.shape + (1, ))  # reshape
    i = 0
    aug_images = []
    for batch in datagen.flow(image, batch_size=1):
        aug_image = batch[0].reshape(image.shape[1:3])  
        hog_features = hog(aug_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        aug_images.append(hog_features)
        i += 1
        if i >= 4:  # generates 4 images
            break
    return aug_images


malignant_images, malignant_labels = load_images_from_folder(path_to_malignant)
benign_images, benign_labels = load_images_from_folder(path_to_benign)


X = np.array(benign_images + malignant_images)
y = np.array(benign_labels + malignant_labels)

augmented_images = []
augmented_labels = []
for img, label in zip(X, y):
    aug_images = augment_image(img)
    augmented_images.extend(aug_images)
    augmented_labels.extend([label] * len(aug_images))

X = np.concatenate((X, np.array(augmented_images)), axis=0) # adds augmented_image
y = np.concatenate((y, np.array(augmented_labels)), axis=0)

print(f"Total number of images: {len(X)}")  # 50000 = 10000 + (4 x 10000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM classifier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Results
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
