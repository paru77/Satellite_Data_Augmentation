import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import Model as Model_keras
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2

nuclear_data=pd.read_csv('/DS/dsg-ml/work/pnegi/CH_nuclear_2023.csv')
# Defining the start date (January 1st, 2023 at 00:00)
start_date = datetime(2023, 1, 1)

# Convert the 'Unnamed: 0' column to timestamps by adding the number of hours to the start_date
nuclear_data['timestamp'] = nuclear_data['Unnamed: 0'].apply(lambda x: start_date + timedelta(hours=x))

# Drop the 'Unnamed: 0' column or rename it if necessary
nuclear_data = nuclear_data.drop(columns=['Unnamed: 0'])

# Optionally reorder the columns to put 'timestamp' at the front
nuclear_data = nuclear_data[['timestamp'] + [col for col in nuclear_data.columns if col != 'timestamp']]

print(nuclear_data.head())
print(nuclear_data.tail())

nuclear_data['timestamp']=pd.to_datetime(nuclear_data['timestamp'])
aggerated_data=nuclear_data.resample('D', on='timestamp').sum()

gosgen_nuclear_data=pd.DataFrame(aggerated_data["Gösgen"])
gosgen_nuclear_data['label']=gosgen_nuclear_data['Gösgen'].apply(lambda x: 1 if x>0 else 0)

# Image input shape
image_shape = (256, 256, 3)  # Example shape for RGB images



# Define the model
model = load_model('multi_branch_nuclear_plant_model.h5')
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


# Defining the data directory paths for each attribute
data_dirs = {
    'thermal': '/DS/dsg-ml/work/pnegi/dataset/gosgen/thermal',
    'natural': '/DS/dsg-ml/work/pnegi/dataset/gosgen/natural',
    'aerosol': '/DS/dsg-ml/work/pnegi/dataset/gosgen/aerosol',
    'optical_thickness': '/DS/dsg-ml/work/pnegi/dataset/gosgen/optical_thickness',
    'moisture': '/DS/dsg-ml/work/pnegi/dataset/gosgen/moisture',
    "chlorophyll":"/DS/dsg-ml/work/pnegi/dataset/gosgen/chlorophyll"
}


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
image_data = {}
for attr, path in data_dirs.items():
    image_data[attr], timestamps = load_images(path)
# Extract labels where the index (timestamp) matches the timestamps list
labels = gosgen_nuclear_data.loc[gosgen_nuclear_data.index.isin(timestamps), "label"].values


# Find the minimum number of samples across all attributes
min_samples = min(data.shape[0] for data in image_data.values())

# Trim all datasets to the minimum number of samples
for attr in image_data.keys():
    image_data[attr] = image_data[attr][:min_samples]

# Trim labels to match the minimum samples
labels = labels[:min_samples]


# Prepare individual inputs for each branch
X = {attr: image_data[attr] for attr in ['thermal', 'natural', 'aerosol', 'optical_thickness', 'moisture', 'chlorophyll']}
y = labels


X_train = {}
X_test = {}
y_test=y_train=None

for attr in X.keys():
    X_train[attr], X_test[attr], y_train, y_test = train_test_split(X[attr], y, test_size=0.2, random_state=42)
    print(f"Attribute: {attr}, Train shape: {X_train[attr].shape}, Test shape: {X_test[attr].shape}")



x_train_final={}
x_val={}
y_train_final=y_val=None

for attr in X_train.keys():
    x_train_final[attr],x_val[attr],y_train_final,y_val=train_test_split(X_train[attr],y_train,test_size=0.2,random_state=42)
    print(f"Attribute: {attr}, Train shape: {x_train_final[attr].shape}, Test shape: {x_val[attr].shape}")




# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    {attr: x_train_final[attr] for attr in X.keys()},  # Input dictionary
    y_train_final,  # Labels
    validation_data=(
        {attr: x_val[attr] for attr in X.keys()},  # Input dictionary for test set
        y_val
    ),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)



# Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    {attr: X_test[attr] for attr in X.keys()},  # Input dictionary for test set
    y_test
)
print(f"Test Accuracy: {test_accuracy:.2f}")

y_pred = model.predict({attr: X_test[attr] for attr in X.keys()})
y_pred_classes = (y_pred > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("plots/heatmap_L_G.png")

# -------------------------------------------------#
# #visualizing which parts of an input image contribute most to your model's predictions

# def compute_grad_cam(model, image, layer_name, class_idx=None):
#     """
#     Compute Grad-CAM for the given model, image, and layer name.
#     """
#     grad_model = Model_keras(
#         inputs=[model.input],
#         outputs=[model.get_layer(layer_name).output, model.output]
#     )
#     with tf.GradientTape() as tape:
#         conv_output, predictions = grad_model([image])
#         if class_idx is None:
#             class_idx = tf.argmax(predictions[0])  # Get the predicted class index
#         class_channel = predictions[:, class_idx]

#     # Compute the gradient of the top predicted class with respect to the convolutional output
#     grads = tape.gradient(class_channel, conv_output)
#     # Compute the mean intensity of the gradient
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     # Multiply each channel by "how important" it is with regard to the class
#     conv_output = conv_output[0]
#     heatmap = tf.reduce_sum(pooled_grads * conv_output, axis=-1)


#     # Normalize the heatmap to a range of 0 to 1 for visualization
#     heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

#     return heatmap

# def overlay_heatmap(heatmap, image, alpha=0.6, colormap=cv2.COLORMAP_JET):
#     """
#     Overlay the heatmap on the image for visualization.
#     """
#     heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, colormap)
#     overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
#     return overlay

# # Example usage with the thermal image branch
# sample_image=X_test["natural"][0]
# sample_image_batch ={"thermal":np.expand_dims(X_test["thermal"][0],axis=0),
# "natural":np.expand_dims(X_test["natural"][0],axis=0),
# "aerosol":np.expand_dims(X_test["aerosol"][0],axis=0),
# "optical_thickness":np.expand_dims(X_test["optical_thickness"][0],axis=0),
# "moisture":np.expand_dims(X_test["moisture"][0],axis=0),
# "chlorophyll":np.expand_dims(X_test["chlorophyll"][0],axis=0)
# }
# layer_name = 'conv2d_2'  # Replace with the actual convolutional layer name from your model

# heatmap = compute_grad_cam(model, sample_image_batch, layer_name)
# overlay = overlay_heatmap(heatmap, (sample_image * 255).astype(np.uint8))  # Denormalize the image for visualization

# # Plot the results
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow((sample_image * 255).astype(np.uint8))  # Denormalize the image
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Grad-CAM Overlay")
# plt.imshow(overlay)
# plt.axis('off')

# plt.show()
# plt.savefig("plots/AttnMap_FirstConvNatural.png")


# # ---------------------------------------------------#end of attention maps

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("plots/accuracyPlotGosgen.png")

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("plots/lossPlotGosgen.png")


model.save('trained_leibstadt_gosgen.h5')