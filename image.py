import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import os

# Data Generators
data_gen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                              shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_data = r"C:\Users\ACER\Desktop\CNN\Training"
train_generator = data_gen.flow_from_directory(train_data, target_size=(32, 32), batch_size=32, class_mode='binary')

valid_gen = ImageDataGenerator(rescale=1./255)
validation_data = r"C:\Users\ACER\Desktop\CNN\Validation"
valid_generator = valid_gen.flow_from_directory(validation_data, target_size=(32, 32), batch_size=32, class_mode='binary')

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
'''
models = model.compile(metrics = ["Accuracy"], loss = "binary_crossentropy",optimizer = "Adam")
model.fit(train_generator , epochs = 6)
# After completing the epochs
model.save("my_models.h5")

'''
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("my_models.h5")



predictions = model.predict(valid_generator)

# Convert the probabilities into class labels
predicted_labels = [1 if prob > 0.5 else 0 for prob in predictions]  # Assuming binary classification with a threshold of 0.5


# Get the filenames and true labels from the valid_generator
filenames = valid_generator.filenames
true_labels = valid_generator.classes  # Assuming binary classification

# Plot some example images along with their predicted and true labels
num_images_to_plot = 10
num_cols = int(num_images_to_plot / 2)  # Calculate the number of columns as an integer
plt.figure(figsize=(10, 5))
# Separate the predictions and true labels for male and female classes
male_predictions = []
female_predictions = []
male_true_labels = []
female_true_labels = []

for i in range(len(predicted_labels)):
    if true_labels[i] == 0:  # Assuming 0 represents the male class
        male_predictions.append(predicted_labels[i])
        male_true_labels.append(true_labels[i])
    else:
        female_predictions.append(predicted_labels[i])
        female_true_labels.append(true_labels[i])

# Plot some example images along with their predicted and true labels for males
plt.figure(figsize=(10, 5))
plt.suptitle("Female Predictions")
for i in range(num_images_to_plot):
    # Load the image
    img_path = os.path.join(validation_data, filenames[i])
    img = plt.imread(img_path)

    # Display the image
    plt.subplot(2, num_cols, i+1)  # Use num_cols instead of num_images_to_plot/2
    plt.imshow(img)
    plt.axis('off')

    # Display the predicted and true labels for males
    plt.title(f'Predicted: {male_predictions[i]}, True: {male_true_labels[i]}')

plt.tight_layout()
plt.show()

# Plot some example images along with their predicted and true labels for females
plt.figure(figsize=(10, 5))
plt.suptitle("Male Predictions")
for i in range(num_images_to_plot):
    # Load the image
    img_path = os.path.join(validation_data, filenames[i+len(male_predictions)])  # Skip male images
    img = plt.imread(img_path)

    # Display the image
    plt.subplot(2, num_cols, i+1)  # Use num_cols instead of num_images_to_plot/2
    plt.imshow(img)
    plt.axis('off')

    # Display the predicted and true labels for females
    plt.title(f'Predicted: {female_predictions[i]}, True: {female_true_labels[i]}')

plt.tight_layout()
plt.show()

