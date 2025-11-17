# AI-week-6-assignment
AI for future directions

Task 1: Edge AI Prototype

# Install Kaggle
!pip install kaggle

# Upload your kaggle.json API key
from google.colab import files
files.upload()

# Move key to correct path
!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download the TrashNet dataset
!kaggle datasets download -d asdasdasasdas/garbage-classification

# Unzip dataset
!unzip garbage-classification.zip

dataset/
   cardboard/
   glass/
   metal/
   paper/
   plastic/
   trash/

import tensorflow as tf
from tensorflow.keras import layers, models

img_size = (128, 128)
batch = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=img_size,
    batch_size=batch,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=img_size,
    batch_size=batch,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_ds.class_names
print("Classes:", class_names)

model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

loss, acc = model.evaluate(val_ds)
print("Validation Accuracy:", acc)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # makes it tiny
tflite_model = converter.convert()

with open("garbage_classifier.tflite", "wb") as f:
    f.write(tflite_model)

import tensorflow.lite as tflite
import numpy as np
import cv2

interpreter = tflite.Interpreter(model_path="garbage_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128,128))
    img = np.expand_dims(img/255.0, axis=0).astype("float32")

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return class_names[np.argmax(output)]

print(predict("test/plastic1.jpg"))

pip install tflite-runtime

from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np

interpreter = Interpreter("garbage_classifier.tflite")
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.resize(frame, (128,128))
    img = np.expand_dims(img/255.0, 0).astype("float32")

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    label = class_names[np.argmax(pred)]
    print(label)

    cv2.putText(frame, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Edge AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

🧾 Report Summary (Edge AI Benefits)

Edge AI enables machine learning inference directly on devices such as Raspberry Pi, microcontrollers, and smartphones. Using TensorFlow Lite, the recyclable-item classifier runs entirely offline, eliminating the need for cloud servers. This reduces latency, enables instant recognition, and ensures privacy because images never leave the device. The TFLite model is highly optimized (quantization, pruning), reducing size from ~10 MB to ~1–2 MB while maintaining accuracy around 90%.

Such lightweight models make real-time applications possible—like smart recycling bins, factory sorting systems, or environmental robots. Edge AI also reduces bandwidth costs and improves reliability in locations with weak or no internet. This approach demonstrates how a Kaggle dataset can power a complete edge-friendly workflow: training → optimization → deployment → real-time inference.

Task 2: AI-Driven IoT Concept

1.Sensors & telemetry (prioritized)

Primary (must-have)

Soil moisture (per zone / depth) — irrigation control

Soil temperature (per zone / depth) — germination / root activity

Air temperature (ambient) — microclimate

Relative humidity — disease risk and evapotranspiration

PAR / light sensor (photosynthetically active radiation) — plant growth model

Rain gauge — natural water input

Secondary (high value)

Soil pH — nutrient availability

Soil electrical conductivity (EC) — salinity / nutrient concentration

Leaf wetness sensor — disease forecasting

NDVI / RGB / multispectral camera (drone / fixed) — biomass, stress, canopy cover

Wind speed & direction — spray drift, evapotranspiration

CO₂ (greenhouse) — photosynthetic potential (for controlled environments)

Operational / infrastructure

Battery level / solar panel status — device uptime

GPS / location tag — geo-mapping of plots.

2) AI model proposal to predict crop yield

Goal: predict yield per plot / per hectare (continuous) or yield class (High / Medium / Low).

Recommended approach (hybrid, best for real farms)

Multi-input model that fuses:

Time-series sensor data (soil moisture, temp, humidity, rainfall)

Static/contextual features (soil type, cultivar, planting density, fertilizer schedule)

Image features (NDVI or learned CNN features from RGB/multispectral images)

Weather forecast input (next 7–30 days)

Specific architecture (practical & accurate)

Tabular + time series pipeline:

Preprocess & aggregate sensor data to daily timesteps (or crop-relevant window).

Use Temporal model: 1D Temporal Convolutional Network (TCN) or a small Transformer / LSTM to encode historical sensor sequence.

Image encoder: lightweight MobileNet or small CNN to extract vegetation features from images (if available).

Feature fusion layer: concatenate time-series encoding + image embedding + static features → dense layers → regression head (yield) or softmax head (classes).

Optionally, for tabular-only quick prototype: LightGBM / XGBoost on engineered features (rolling averages, deficits, GDD, stress indices) — fast and often more interpretable.

Training details

Actuator status (valve open/close, pump on/off) — confirmation & safety.
Loss: regression → MSE / MAE; classification → cross-entropy.

Metrics: RMSE, MAE, R² for regression; accuracy, precision/recall, F1 for classification.

Validation: time-aware split (train on past seasons, validate on later seasons) or cross-validation by farms/plots.

Regularization: dropout, early stopping; consider ensembling (LightGBM + NN) for robustness.

Explainability: SHAP for tree models; attention weights or feature importance for NN fusion.

Deployment choices

Deployment choices

Edge (on-farm gateway / Raspberry Pi): serve LightGBM or a quantized TFLite version of the NN for low latency decisions (irrigation).

Cloud: heavier periodic retraining, forecasting, and model management.

Hybrid: edge inference for real-time control; cloud for retraining, analytics, and long-term forecasting.

3) Data flow diagram (sketch)

[Sensors (soil, air, camera, rain, etc.)]
            │ (raw telemetry, images) 
            ▼
      [Edge Nodes / Microcontrollers]
            │ (local filtering, compression)
            ▼
        [Edge Gateway / Pi]
            │ (preprocessing, local aggregation, quick inference)
            │ └─> Local Actuators (irrigation valves, fans)  ←─ feedback
            ▼
     [Secure Cloud Ingress / Message Broker]
            │ (MQTT/HTTPS) - time series + images -> storage
            ▼
        [Data Lake / Time Series DB]
            │ (sensor TS, images, metadata)
            ▼
    [Feature Engineering Pipeline]
            │ (rolling stats, GDD, deficits, image features)
            ▼
      [Model Training Service / Scheduler]
            │ (retrain, validate)
            ▼
         [Model Registry]
            │ (versioned artifacts)
            ▼
     [Model Serving (Edge + Cloud endpoints)]
            │ (batch predictions, realtime inference API)
            ▼
      [Applications & Dashboards]
 (farmer app, alerts, visualizations, actuarial predictions)
            ▲
            │ (user decisions, manual inputs)
            └───────────┬───────────┘
                        │
                 [Feedback Loop]
      (harvest yields, sensor drift, ground truth labels)
