import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def load_images_from_folder(folder, image_size=(64, 64), max_images=5000):
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if i >= max_images: #ensures 25000 images (1 + 1 x 4) x 5000
            break
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).resize(image_size)
        img = np.array(img)
        if img.shape == image_size + (3,): # verifies image size
            images.append(img)
    return np.array(images)

benign_folder = "direct path to benign"
malignant_folder = "direct path to malignant"

benign_images = load_images_from_folder(benign_folder)
malignant_images = load_images_from_folder(malignant_folder)

benign_labels = np.zeros(len(benign_images))
malignant_labels = np.ones(len(malignant_images))

images = np.concatenate((benign_images, malignant_images), axis=0)
labels = np.concatenate((benign_labels, malignant_labels), axis=0)

# 80-20 train test split from sklearn
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# use ImageDataGenerator for augmentation
# same datagen parameters as SVM
datagen = ImageDataGenerator(
    rotation_range=20, #rotate
    width_shift_range=0.1, #horizontal shift
    height_shift_range=0.1, #vertical shift
    shear_range=0.1, #shear
    zoom_range=0.1, #scale
    horizontal_flip=True, #flip
    fill_mode='nearest'
)

augmented_images = []
augmented_labels = []

# append augmented
for img, label in zip(X_train, y_train):
    img = img.reshape((1,) + img.shape)
    i = 0
    for batch in datagen.flow(img, batch_size=1): #passes through datagen variable defiend above
        augmented_images.append(batch[0])
        augmented_labels.append(label)
        i += 1
        if i > 5:  # 4 images
            break

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# adds augmented images and labels to original dat
X_train = np.concatenate((X_train, augmented_images), axis=0)
y_train = np.concatenate((y_train, augmented_labels), axis=0)

print(f"# of images: {len(X_train)}") # 50000 = 10000 + (4 x 10000)

X_train = (X_train - 127.5) / 127.5 # standardize
X_test = (X_test - 127.5) / 127.5

# generator architecture
def generator():
    model = Sequential()
    model.add(Dense(256 * 16 * 16, activation="relu", input_dim=100))
    model.add(Reshape((16, 16, 256)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=4, padding="same", activation='tanh'))
    return model

# discriminator architecture
def discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=(64, 64, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# build discriminator
discriminator = discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# build generator
generator = generator()

z = tf.keras.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)

gan = tf.keras.Model(z, valid) # defines gan as combination of generator and discriminator
gan.compile(loss='binary_crossentropy', optimizer='adam')

def train_gan(generator, discriminator, gan, epochs, batch_size=32, save_interval=50):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100)) #random noise for generator
        gen_imgs = generator.predict(noise)

        loss_real = discriminator.train_on_batch(real_imgs, valid)
        loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        discr_loss = 0.5 * np.add(loss_real, loss_fake)

      
        g_loss = gan.train_on_batch(noise, valid)

        # intermittent progress saving
        if epoch % save_interval == 0:
            print(f"{epoch} [discriminator loss: {d_loss[0]} | discriminator accuracy: {100*discr_loss[1]}] [generator loss: {g_loss}]")
            save_images(generator, epoch)

def save_images(generator, epoch, examples=3, dim=(1, 3), figsize=(3, 3)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise) 
    generated_images = 0.5 * generated_images + 0.5  
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

# 1000 epochs, 32 batch size
train_gan(generator, discriminator, gan, epochs=1000, batch_size=32, save_interval=200)

# evaluate discriminator!
loss, accuracy = discriminator.evaluate(X_test, y_test)
print(f"Discriminator test accuracy: {accuracy}")
