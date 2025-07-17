import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import seaborn as sns
import cv2
from PIL import Image
import numpy as np
import os



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


#checking image is empty or not
def imageEmpty(data_dir):
    total_images = 0
    useful_images = 0
    for file in sorted(os.listdir(data_dir)):
        if file.endswith('.PNG'):
            img = Image.open(os.path.join(data_dir, file))
            total_images+=1
            # print(f"Image size for {file}: {img.size}")
            #checking if image is entirely transparent-opticalThickness,moisture"
            if img.mode == 'RGBA':
                alpha_channel = np.array(img.getchannel("A"))
                transparent_ratio = np.sum(alpha_channel == 0) / alpha_channel.size
                if transparent_ratio > 0.9:
                    continue;
                # if np.all(alpha_channel == 0):
                #   print(f"Image {file} : is empty")

            
            img_gray = img.convert("L")
            img_array = np.array(img_gray)

            # Check if the image is entirely black, natural
            if np.all(img_array == 0):
                # print(f"Image {file} : is empty")
                continue;
            
            useful_images += 1
    return useful_images, total_images
            
useful_data = {}
for key, path in data_dirs_leibstadt.items():
    useful, total = imageEmpty(path)
    useful_data[key] = (useful, total)

for key, path in data_dirs_gosgen.items():
    useful, total = imageEmpty(path)
    newval,newtotal=useful+useful_data[key][0],total+useful_data[key][1]
    useful_data[key]=(newval,newtotal)



# Plotting the results
attributes = list(useful_data.keys())
percentages = [(useful_data[attr][0] / useful_data[attr][1]) * 100 for attr in attributes]

print(attributes)
print(percentages)
# Ensure the output directory exists
output_dir = "plots/tests"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.bar(attributes, percentages, color='skyblue')
plt.xlabel("Attributes")
plt.ylabel("Percentage of Useful Images (%)")
plt.title("Percentage of Useful Data by Attribute")

# Save the plot before showing it
save_path = os.path.join(output_dir, "TotalUseFullData.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Ensure high-quality save
plt.close()  # Close figure to free memory

print(f"Plot saved successfully at: {save_path}")