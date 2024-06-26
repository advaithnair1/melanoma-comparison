import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import hog

# Paths to the data folders
path_to_benign = "direct path to benign"
path_to_malignant = "direct path to malignant"

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((300, 300))  # Resize to 300x300
            img_array = np.array(img)  # Convert to numpy array
            hog_features = hog(img_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)  # HOG for feature extraction; better than LBP and Gabor
            images.append(hog_features)
            label = 1 if 'malignant' in folder else 0  # Label: 1 for malignant, 0 for benign
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels

def augment_image(image):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator( #datagen variable for augmentation
        rotation_range=20, #rotate
        width_shift_range=0.1, #horizontal shift
        height_shift_range=0.1, #vertical shift
        shear_range=0.1, #shears
        zoom_range=0.1, #scales
        horizontal_flip=True, #flips
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

# adds augmented images and labels to existing dataset
augmented_images = []
augmented_labels = []
for img, label in zip(X, y):
    aug_images = augment_image(img)
    augmented_images.extend(aug_images)
    augmented_labels.extend([label] * len(aug_images))

X = np.concatenate((X, np.array(augmented_images)), axis=0) # adds augmented_image
y = np.concatenate((y, np.array(augmented_labels)), axis=0)

print(f"# of images: {len(X)}")  # 50000 = 10000 + (4 x 10000)

# 80-20 train test split from sklearn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# base SVM model form sklearn
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))

# conf matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
