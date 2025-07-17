import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import seaborn as sns


nuclear_data=pd.read_csv('/DS/dsg-ml/work/pnegi/CH_nuclear_2023.csv')
# Defining the start date (January 1st, 2023 at 00:00)
start_date = datetime(2023, 1, 1)

# Convert the 'Unnamed: 0' column to timestamps by adding the number of hours to the start_date
nuclear_data['timestamp'] = nuclear_data['Unnamed: 0'].apply(lambda x: start_date + timedelta(hours=x))

# Drop the 'Unnamed: 0' column or rename it if necessary
nuclear_data = nuclear_data.drop(columns=['Unnamed: 0'])

# Optionally reorder the columns to put 'timestamp' at the front
nuclear_data = nuclear_data[['timestamp'] + [col for col in nuclear_data.columns if col != 'timestamp']]


nuclear_data['timestamp']=pd.to_datetime(nuclear_data['timestamp'])
aggerated_data=nuclear_data.resample('D', on='timestamp').sum()

leibstadt_nuclear_data= pd.DataFrame(aggerated_data['Leibstadt'])
leibstadt_nuclear_data['label_Leibstadt']=leibstadt_nuclear_data['Leibstadt'].apply(lambda x: 1 if x>0 else 0)

print(leibstadt_nuclear_data.head())

gosgen_nuclear_data=pd.DataFrame(aggerated_data['Gösgen'])
gosgen_nuclear_data['label_Gosgen']=gosgen_nuclear_data['Gösgen'].apply(lambda x: 1 if x>0 else 0)

print(gosgen_nuclear_data.head())

data_dirs_leibstadt = {
    'thermal': '/DS/dsg-ml/work/pnegi/dataset/leibstadt/thermal',
    'natural': '/DS/dsg-ml/work/pnegi/dataset/leibstadt/natural',
    'optical_thickness': '/DS/dsg-ml/work/pnegi/dataset/leibstadt/optical_thickness',
    'moisture': '/DS/dsg-ml/work/pnegi/dataset/leibstadt/moisture',
    "chlorophyll":"/DS/dsg-ml/work/pnegi/dataset/leibstadt/chlorophyll"
}

data_dirs_gosgen = {
    'thermal': '/DS/dsg-ml/work/pnegi/dataset/gosgen/thermal',
    'natural': '/DS/dsg-ml/work/pnegi/dataset/gosgen/natural',
    'optical_thickness': '/DS/dsg-ml/work/pnegi/dataset/gosgen/optical_thickness',
    'moisture': '/DS/dsg-ml/work/pnegi/dataset/gosgen/moisture',
    "chlorophyll":"/DS/dsg-ml/work/pnegi/dataset/gosgen/chlorophyll"
}

# Adjust the shape to include all attributes as channels
combined_channel_shape = (256, 256, len(data_dirs_leibstadt)*3)

# Define a single CNN model
def create_unified_cnn(input_shape):
    input_layer = Input(shape=input_shape, name="combined_input")
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs=input_layer, outputs=output)
    return model


def load_images(data_dir, target_size=(256,256)):
    images = []
    timestamps = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".PNG"):
            timestamps.append(file.split("_")[0])  # Assuming timestamp in filename
            img = load_img(os.path.join(data_dir, file), target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
    return np.array(images), timestamps

# Load data for all attributes
image_data_leibstadt = {}
timestamps_lib=[]
for attr, path in data_dirs_leibstadt.items():
    image_data_leibstadt[attr], timestamps_lib = load_images(path)

labels_lib = leibstadt_nuclear_data.loc[leibstadt_nuclear_data.index.isin(timestamps_lib), "label_Leibstadt"].values    
min_samples=min(data.shape[0] for data in image_data_leibstadt.values())

for attr in image_data_leibstadt.keys():
    image_data_leibstadt[attr] = image_data_leibstadt[attr][:min_samples]

labels_lib=labels_lib[:min_samples]


image_data_gosgen={}
timestamps_gos=[]
for attr, path in data_dirs_gosgen.items():
    image_data_gosgen[attr],timestamps_gos=load_images(path)
 
labels_gos = gosgen_nuclear_data.loc[gosgen_nuclear_data.index.isin(timestamps_gos), "label_Gosgen"].values    
min_samples=min(data.shape[0] for data in image_data_gosgen.values())

for attr in image_data_gosgen.keys():
    image_data_gosgen[attr] = image_data_gosgen[attr][:min_samples]
    
labels_gos=labels_gos[:min_samples]

combined_images = {attr: np.concatenate([
    image_data_leibstadt[attr], image_data_gosgen[attr]
], axis=0) for attr in image_data_leibstadt.keys()}


# Combine all attributes into a single array per image
def combine_attributes(image_data):
    combined_images = np.concatenate([image_data[attr] for attr in image_data.keys()], axis=-1)
    return combined_images


Combined_images_data =combine_attributes(combined_images)

print(f"shape of images after stacking :{Combined_images_data.shape}")

combined_labels=np.concatenate([labels_lib,labels_gos],axis=0)
print(f"Labels shape: {combined_labels.shape}")

# Train/test/validation split
X_train, X_temp, y_train, y_temp = train_test_split(Combined_images_data, combined_labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# --- Augmentation for 'off' class (label 0) ---
# Find indices of 'off' samples in the training set
off_indices = np.where(y_train == 0)[0]
X_train_off = X_train[off_indices]
y_train_off = y_train[off_indices]

# Find indices of 'on' samples in the training set
on_indices = np.where(y_train == 1)[0]

# Define the ImageDataGenerator for augmentation
# You can customize these parameters based on your data and desired augmentation types
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Determine how many augmented samples to generate for the 'off' class
# Goal: Make the number of 'off' samples closer to or equal to 'on' samples
num_on_samples = len(on_indices)
num_off_samples = len(y_train_off)
print(f"Original 'on' samples in training: {num_on_samples}")
print(f"Original 'off' samples in training: {num_off_samples}")

if num_off_samples > 0:
    target_augmented_off_samples = num_off_samples * 3 # Example: at least as many as 'on', or double 'off'
    
    augmented_X_off = []
    augmented_y_off = []
    
    # Using flow to generate augmented batches
    i = 0
    for batch_X, batch_y in datagen.flow(X_train_off, y_train_off, batch_size=num_off_samples, shuffle=False):
        augmented_X_off.append(batch_X)
        augmented_y_off.append(batch_y)
        i += len(batch_X)
        if i >= (target_augmented_off_samples): # Generate enough to reach target, accounting for original
            break
    
    augmented_X_off = np.concatenate(augmented_X_off, axis=0)
    augmented_y_off = np.concatenate(augmented_y_off, axis=0)

    print(f"Generated {len(augmented_y_off)} augmented 'off' samples.")

    # Combine original 'on' samples with original 'off' samples and augmented 'off' samples
    X_train_augmented = np.concatenate([X_train, augmented_X_off], axis=0)
    y_train_augmented = np.concatenate([y_train, augmented_y_off], axis=0)
else:
    print("No 'off' samples found in training data, skipping augmentation.")
    X_train_augmented = X_train
    y_train_augmented = y_train

# Shuffle the augmented training data
shuffle_indices = np.arange(len(y_train_augmented))
np.random.shuffle(shuffle_indices)
X_train_augmented = X_train_augmented[shuffle_indices]
y_train_augmented = y_train_augmented[shuffle_indices]

print(f"New training data shape after augmentation: {X_train_augmented.shape}")
print(f"New training labels shape after augmentation: {y_train_augmented.shape}")



# Adjust the shape to include all attributes as channels
combined_channel_shape = (256, 256, len(data_dirs_leibstadt)*3)

# Build and compile the model
unified_model = create_unified_cnn(combined_channel_shape)
unified_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name='auc')])
unified_model.summary()

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_augmented),
    y=y_train_augmented
)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights_dict)

# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = unified_model.fit(
    X_train_augmented, y_train_augmented,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
    class_weight=class_weights_dict
)


# Evaluate on test data
test_loss, test_accuracy,AUC = unified_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# New part: Analyzing impact of removing each feature
features = list(combined_images.keys())
print(features)
# First, get the baseline accuracy (already evaluated, but re-doing here to be clean)
full_test_loss, full_test_accuracy, AUC = unified_model.evaluate(X_test, y_test)
print(f"Full Feature Test Accuracy: {full_test_accuracy:.4f}")

# Now, test accuracy when each feature is missing
channels_per_feature = 3  # because RGB images

feature_accuracies = {}

for idx, feature in enumerate(features):
    print(f"Evaluating without feature: {feature}")

    # Copy X_test so original isn't modified
    X_test_modified = np.copy(X_test)

    # Set corresponding channels to zero
    start_channel = idx * channels_per_feature
    end_channel = (idx + 1) * channels_per_feature

    X_test_modified[:, :, :, start_channel:end_channel] = 0  # Set channels to 0

        # Evaluate
    loss, accuracy, AUC = unified_model.evaluate(X_test_modified, y_test, verbose=0)
    feature_accuracies[feature] = accuracy

# Now plot the accuracy drops
accuracy_drops = {feat: full_test_accuracy - acc for feat, acc in feature_accuracies.items()}

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(accuracy_drops.keys(), accuracy_drops.values(), color='salmon')
plt.xlabel('Feature Removed',fontsize=16)
plt.ylabel('Accuracy Drop',fontsize=16)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("plots/Final_Exp/accuracy_drop_per_feature.png")

# Print drops nicely
for feat, drop in accuracy_drops.items():
    print(f"Removing {feat}: Accuracy drop of {drop:.4f}")
