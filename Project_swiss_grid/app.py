from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
with tf.device("/CPU:0"):
    model = load_model('/app/Unified_model.keras')

@app.route('/')
def home():
    return "Fast API running"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Log the incoming request
        print("Received request:", request.json)

        data = request.json  # Expect JSON payload 
        images=data['images']
        if not data or 'images' not in data:
            return jsonify({'error': 'Invalid input. JSON with key "images" is required.'}), 400

        if len(images)!=5:
            return jsonify({"error": f"Exactly 5 images required, Received {len(images)}."}),400

        preprocessed_images = []
        target_size=(256,256)
       
        for image in data['images']:
            img = load_img(image, target_size=target_size)
            img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            preprocessed_images.append(img_array)
        combined_images = np.concatenate(preprocessed_images, axis=-1)

        # Make predictions
        predictions = model.predict(np.expand_dims(combined_images, axis=0))  # Add batch dimension
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)




# #sentinel data
#     kernel_planckster, protocol, file_repository = setup(
#                 job_id=job_id,
#                 logger=logger,
#                 kp_auth_token=kp_auth_token,
#                 kp_host=kp_host,
#                 kp_port=kp_port,
#                 kp_scheme=kp_scheme,
#             )

#     scraped_data_repository = ScrapedDataRepository(
#                 protocol=protocol,
#                 kernel_planckster=kernel_planckster,
#                 file_repository=file_repository,
#             )

    # data = request.json  # Expect JSON payload
    # images = np.array(data['images'])  # Input images as a NumPy array
    # predictions = model.predict(images)
    # return jsonify({'predictions': predictions.tolist()})