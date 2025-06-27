import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

# Function to load and preprocess images from directories
def load_images_from_directories(directories, target_size=(128, 128)):
    images = []
    labels = []
    label_map = {name: idx for idx, name in enumerate(directories)}

    for label, directory in directories.items():
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    image_path = os.path.join(root, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, target_size)
                    images.append(image)
                    labels.append(label_map[label])
                
    return np.array(images), np.array(labels)

# Define paths to training directories
train_directories = {
    "glioma_tumor": "Training_Sample/glioma_tumor",
    "meningioma_tumor": "Training_Sample/meningioma_tumor",
    "no_tumor": "Training_Sample/no_tumor",
    "pituitary_tumor": "Training_Sample/pituitary_tumor"
}

# Define paths to test directories
test_directories = {
    "glioma_tumor": "Brain_Tumor_Detection/Testing/glioma_tumor",
    "meningioma_tumor": "Brain_Tumor_Detection/Testing/meningioma_tumor",
    "no_tumor": "Brain_Tumor_Detection/Testing/no_tumor",
    "pituitary_tumor": "Brain_Tumor_Detection/Testing/pituitary_tumor"
}

# Load training images from directories
X, y = load_images_from_directories(train_directories)

# Load test images from directories
X_test, y_test = load_images_from_directories(test_directories)

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val)

# Expand dimensions for CNN input
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define the U-Net Architecture with more filters
def unet_model(input_shape=(128, 128, 1)):
    inputs = Input(input_shape)
    c1 = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, (5, 5), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (5, 5), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, (5, 5), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (5, 5), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, (5, 5), activation='relu', padding='same')(c3)

    u2 = UpSampling2D((2, 2))(c3)
    c4 = Conv2D(128, (5, 5), activation='relu', padding='same')(u2)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(128, (5, 5), activation='relu', padding='same')(c4)

    u3 = UpSampling2D((2, 2))(c4)
    c5 = Conv2D(64, (5, 5), activation='relu', padding='same')(u3)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(64, (5, 5), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    return model

# Define the Dense Model
def dense_model(input_shape):
    inputs = Input(input_shape)
    x = Flatten()(inputs)
    x = Dense(units=512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(units=256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(units=4, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

# Combine U-Net encoder, Dense model, and Dense classifier
def ensemble_model(unet_input_shape, dense_input_shape):
    unet_encoder = unet_model(input_shape=unet_input_shape)
    dense = dense_model(input_shape=dense_input_shape)

    # Get output layers of both models
    unet_output = unet_encoder.output
    dense_output = dense.output

    # Flatten the spatial dimensions of the U-Net encoder output
    flat_unet_output = Flatten()(unet_output)

    # Concatenate the flattened U-Net encoder output and dense model output
    concatenated = concatenate([flat_unet_output, dense_output])

    # Additional dense layers
    x = Dense(units=64, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    outputs = Dense(units=4, activation='softmax')(x)

    model = Model(inputs=[unet_encoder.input, dense.input], outputs=outputs)
    return model

# Define shapes for U-Net and Dense model inputs
unet_input_shape = (128, 128, 1)
dense_input_shape = X_train[0].shape

# Create the ensemble model
ensemble_model = ensemble_model(unet_input_shape, dense_input_shape)

# Instantiate the optimizer and compile the model
ensemble_model.compile(optimizer=tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    weight_decay=0.01,
    amsgrad=True
), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, min_lr=0.00001)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# Train the model
history = ensemble_model.fit(
    [X_train, X_train], y_train,
    validation_data=([X_val, X_val], y_val),
    epochs=70,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the trained model on the test set
test_loss, test_accuracy = ensemble_model.evaluate([X_test, X_test], y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Function to draw bounding boxes and classify tumors
def predict_and_visualize(model, test_images, test_labels, class_names, target_size=(128, 128)):
    for i in range(len(test_images)):
        # Preprocess the test image
        original_image = test_images[i]
        resized_image = cv2.resize(original_image, target_size)
        input_image = np.expand_dims(resized_image, axis=(0, -1))  # Add batch and channel dimensions

        # Predict tumor class
        predictions = model.predict([input_image, input_image])
        predicted_class = np.argmax(predictions)
        predicted_label = class_names[predicted_class]

        # Generate a bounding box (for simplicity, assume the entire image contains the tumor)
        x, y, w, h = 10, 10, target_size[0] - 20, target_size[1] - 20

        # Visualize the image with the bounding box and label
        display_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(display_image, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the image with bounding box
        plt.figure(figsize=(5, 5))
        plt.imshow(display_image, cmap='gray')
        plt.title(f"Predicted: {predicted_label}, True: {class_names[test_labels[i]]}")
        plt.axis("off")
        plt.show()

# Define the class names for the tumor types
class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Run the function on test images
predict_and_visualize(ensemble_model, X_test, y_test, class_names)
 