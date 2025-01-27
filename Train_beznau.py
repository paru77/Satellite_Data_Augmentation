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

beznau_nuclear_plant=pd.DataFrame(aggerated_data["Beznau_1"])
beznau_nuclear_plant["Beznau_2"]=aggerated_data["Beznau_2"]

beznau_nuclear_plant["label_1"]=beznau_nuclear_plant["Beznau_1"].apply(lambda x: 1 if x>0 else 0)
beznau_nuclear_plant['label_2']=beznau_nuclear_plant["Beznau_2"].apply(lambda x: 1 if x>0 else 0)

print(beznau_nuclear_plant.head())
print(beznau_nuclear_plant.tail())

data_dirs_beznau = {
    'thermal': '/DS/dsg-ml/work/pnegi/dataset/beznau/thermal',
    'natural': '/DS/dsg-ml/work/pnegi/dataset/beznau/natural',
    'optical_thickness': '/DS/dsg-ml/work/pnegi/dataset/beznau/optical_thickness',
    'moisture': '/DS/dsg-ml/work/pnegi/dataset/beznau/moisture',
    "chlorophyll":"/DS/dsg-ml/work/pnegi/dataset/beznau/chlorophyll"
}

# Adjust the shape to include all attributes as channels
combined_channel_shape = (256, 256, len(data_dirs_beznau)*3)


# Define a single CNN model
def create_unified_cnn(input_shape):
    input_layer = Input(shape=input_shape, name="combined_input")
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    
    reactor1_output= Dense(128, activation="relu")(x)
    reactor1_output=Dense(1,activation="sigmoid",name="reactor1_output")(reactor1_output)
    
    reactor2_output= Dense(128, activation="relu")(x)
    reactor2_output=Dense(1,activation="sigmoid",name="reactor2_output")(reactor2_output)
    model = Model(inputs=input_layer, outputs=[reactor1_output,reactor2_output])
    return model

# Build and compile the model
unified_model = create_unified_cnn(combined_channel_shape)
unified_model = create_unified_cnn(combined_channel_shape)
unified_model.compile(optimizer="adam", loss={"reactor1_output":"binary_crossentropy","reactor2_output":"binary_crossentropy"},
 metrics={"reactor1_output":"accuracy","reactor2_output":"accuracy"})
unified_model.summary()

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
image_data_beznau = {}
timestamps_beznau=[]
for attr, path in data_dirs_beznau.items():
    image_data_beznau[attr], timestamps_beznau = load_images(path)

label_1=beznau_nuclear_plant.loc[beznau_nuclear_plant.index.isin(timestamps_beznau), "label_1"].values
label_2=beznau_nuclear_plant.loc[beznau_nuclear_plant.index.isin(timestamps_beznau), "label_2"].values
# print(f"label 1 looks like :{label_1}")

min_samples=min(data.shape[0] for data in image_data_beznau.values())

for attr in image_data_beznau.keys():
    image_data_beznau[attr] = image_data_beznau[attr][:min_samples]
    print(f"shape of each attribute: {image_data_beznau[attr].shape}")

label_1=label_1[:min_samples]
label_2=label_2[:min_samples]


combined_images = np.concatenate([image_data_beznau[attr] for attr in image_data_beznau.keys()], axis=-1)
print(f"shape of images after stacking :{combined_images.shape}")

X_train, X_temp, y_train_1, y_temp_1,y_train_2,y_temp_2 = train_test_split(combined_images,label_1,label_2, test_size=0.3, random_state=42)
X_val, X_test, y_val_1, y_test_1, y_val_2,y_test_2 = train_test_split(X_temp, y_temp_1,y_temp_2, test_size=0.5, random_state=42)


# Training labels
y_train = {"reactor1_output": y_train_1, "reactor2_output": y_train_2}

# Validation labels
y_val = {"reactor1_output": y_val_1, "reactor2_output": y_val_2}

# Test labels
y_test = {"reactor1_output": y_test_1, "reactor2_output": y_test_2}

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history=unified_model.fit(x=X_train,y=y_train,
        validation_data=(X_val,y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
        )


# Evaluate on test data
test_results = unified_model.evaluate(x=X_test,y=y_test)

test_loss=test_results[0]
reactor_1_loss=test_results[1]
reactor_2_loss=test_results[2]
reactor_1_acc=test_results[3]
reactor_2_acc=test_results[4]
print(f"Total test loss : {test_loss}")
print(f"Reactor 1 loss : {reactor_1_loss}")
print(f"Reactor 2 loss : {reactor_2_loss}")
print(f"Reactor 1 Accuracy : {reactor_1_acc}")
print(f"Reactor 2 Accuracy : {reactor_2_acc}")

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['reactor1_output_accuracy'], label='Reactor 1 Train Accuracy')
plt.plot(history.history['val_reactor1_output_accuracy'], label='reactor 1 Validation Accuracy')
plt.plot(history.history['reactor2_output_accuracy'], label='Reactor 2 Train Accuracy')
plt.plot(history.history['val_reactor2_output_accuracy'], label='reactor 2 Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("plots/Beznau_Train/accuracyPlot.png")

# Plot loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("plots/Beznau_Train/lossPlot.png")

# Save model
unified_model.save('Unified_Beznau_model.keras')
