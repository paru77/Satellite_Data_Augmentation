from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model('Unified_Beznau_model.keras')
model.summary()

data=[
    "/DS/dsg-ml/work/pnegi/dataset/beznau/thermal/2023-01-01_thermal.PNG",
    "/DS/dsg-ml/work/pnegi/dataset/beznau/natural/2023-01-01_natural.PNG",
    "/DS/dsg-ml/work/pnegi/dataset/beznau/optical_thickness/2023-01-01_optical_thickness.PNG",
    "/DS/dsg-ml/work/pnegi/dataset/beznau/moisture/2023-01-01_moisture.PNG",
    "/DS/dsg-ml/work/pnegi/dataset/beznau/chlorophyll/2023-01-01_chlorophyll.PNG"
  ]
preprocessed_images = []
target_size=(256,256)

for image in data:
            img = load_img(image, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            preprocessed_images.append(img_array)
combined_images = np.concatenate(preprocessed_images, axis=-1)

# Make predictions
predictions = model.predict(np.expand_dims(combined_images, axis=0)) # Add batch dimension
print(predictions)
predictions = [pred.tolist() for pred in predictions]  
print(f'predictions: {predictions}')