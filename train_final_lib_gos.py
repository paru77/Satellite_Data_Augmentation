#Training model from scratch for gosgen and leibstadt data without aerosol levels as it is 
#just adding noise and not contributing to the predictions
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import Model as Model
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


nuclear_data['timestamp']=pd.to_datetime(nuclear_data['timestamp'])
aggerated_data=nuclear_data.resample('D', on='timestamp').sum()

leibstadt_nuclear_data= pd.DataFrame(aggerated_data['Leibstadt'])
leibstadt_nuclear_data['label_Leibstadt']=leibstadt_nuclear_data['Leibstadt'].apply(lambda x: 1 if x>0 else 0)

print(leibstadt_nuclear_data.head())

gosgen_nuclear_data=pd.DataFrame(aggerated_data['Gösgen'])
gosgen_nuclear_data['label_Gosgen']=gosgen_nuclear_data['Gösgen'].apply(lambda x: 1 if x>0 else 0)

print(gosgen_nuclear_data.head())

# Image input shape
image_shape = (256, 256, 3)
def create_cnn_branch(input_shape, name):
    input_layer = Input(shape=input_shape, name=f"{name}")
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    return input_layer, x

branches = []
inputs = []
for attribute in ['thermal', 'natural', 'optical_thickness', 'moisture', 'chlorophyll']:
    input_layer, branch = create_cnn_branch(image_shape, attribute)
    branches.append(branch)
    inputs.append(input_layer)

# Combine branches
combined = concatenate(branches)
x = Dense(128, activation="relu")(combined)
# x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid", name="output")(x)

# Define the model
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


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
# print(f"min values for lib:{min_samples}")
# Trim all datasets to the minimum number of samples
for attr in image_data_leibstadt.keys():
    image_data_leibstadt[attr] = image_data_leibstadt[attr][:min_samples]

labels_lib=labels_lib[:min_samples]
# print(f"shape for lib labels:{labels_lib.shape}")


image_data_gosgen={}
timestamps_gos=[]
for attr, path in data_dirs_gosgen.items():
    image_data_gosgen[attr],timestamps_gos=load_images(path)
 
labels_gos = gosgen_nuclear_data.loc[gosgen_nuclear_data.index.isin(timestamps_gos), "label_Gosgen"].values    

min_samples=min(data.shape[0] for data in image_data_gosgen.values())
# print(f"Mini sample for gos: {min_samples}")
# Trim all datasets to the minimum number of samples
for attr in image_data_gosgen.keys():
    image_data_gosgen[attr] = image_data_gosgen[attr][:min_samples]
    

labels_gos=labels_gos[:min_samples]
# print(f"labels for gos shape: {labels_gos.shape}")



# Combine satellite image data from both datasets
combined_image_data = {attr: np.concatenate([
    image_data_leibstadt[attr], image_data_gosgen[attr]
], axis=0) for attr in image_data_leibstadt.keys()}

for attr in combined_image_data.keys():
    print(f"shape of {attr}= {combined_image_data[attr].shape}")

combined_labels=np.concatenate([labels_lib,labels_gos],axis=0)
print(f"Labels shape: {combined_labels.shape}")


X_combined={attr:combined_image_data[attr] for attr in combined_image_data.keys()}
y_combined=combined_labels

X_train={}
X_temp={}
X_val={}
X_test={}
y_train,y_test,y_val,y_temp=None,None,None,None

for attr in X_combined.keys():
    X_train[attr],X_temp[attr],y_train,y_temp=train_test_split(X_combined[attr],y_combined,test_size=0.3,random_state=42)
    X_val[attr],X_test[attr],y_val,y_test=train_test_split(X_temp[attr],y_temp,test_size=0.5,random_state=42)
    print(f"Attribute: {attr}, Train shape: {X_train[attr].shape}, Validation shape: {X_val[attr].shape}, Test shape: {X_test[attr].shape}")

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Train the model
history = model.fit(
    {attr: X_train[attr] for attr in X_combined.keys()},  # Input dictionary
    y_train,  # Labels
    validation_data=(
        {attr: X_val[attr] for attr in X_combined.keys()},  # Input dictionary for test set
        y_val
    ),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)



# Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    {attr: X_test[attr] for attr in X_combined.keys()},  # Input dictionary for test set
    y_test
)
print(f"Test Accuracy: {test_accuracy:.2f}")

y_pred = model.predict({attr: X_test[attr] for attr in X_combined.keys()})
y_pred_classes = (y_pred > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d")
plt.savefig("plots/Final_train/heatmap_L_G.png")

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig("plots/Final_train/accuracyPlotGosgen.png")

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig("plots/Final_train/lossPlotGosgen.png")


model.save('Final_train_leibstadt_gosgen.h5')

