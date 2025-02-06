Overview:

This project leverages satellite imagery and deep learning to monitor nuclear power plants in Switzerland. Using a CNN-based model, the system processes multi-band satellite images to detect anomalies in power plant operations and determine whether a plant is active or inactive.

Objectives:

Extract and preprocess satellite images of nuclear power plants.

Utilize multiple spectral bands (thermal, natural, moisture, NO₂, etc.) for analysis.

Develop a CNN-based model to detect anomalies in power generation patterns.

Predict whether a nuclear power plant is operational based on satellite data

Data Sources:

Satellite Imagery: Extracted from Sentinel Hub API, covering thermal, moisture, NO₂, and other relevant bands.

Power Production Data: Aggregated nuclear power generation data for benchmarking predictions.

Model Architecture:

CNN Backbone: MobileNetV2 (or other CNNs) adapted for multi-band input.

Custom Layers: Designed to handle combined spectral image data.

Training Strategy: Supervised learning with binary classification (Plant ON/OFF) and anomaly detection.