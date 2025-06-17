#  Image Classification of Natural and Man-made Scenes

This deep learning project classifies images into six scene categories: **Glacier, Sea, Street, Mountain, Building, Forest** using **Transfer Learning** with EfficientNetB0 or MobileNet as the base model.

---

##  Overview

- **Goal:** Build a multi-class image classification model for natural and urban scenes.
- **Classes:** Glacier, Sea, Street, Mountain, Building, Forest
- **Model Type:** Convolutional Neural Network with Transfer Learning (EfficientNetB0 / MobileNet)
- **Framework:** TensorFlow / Keras

---

##  Architecture

- **Base model:** `EfficientNetB0` or `MobileNet` with pre-trained ImageNet weights.
- **Head:** 
  - GlobalAveragePooling
  - Dense layer with ReLU activation
  - Output layer: `Dense(6, activation='softmax')`

```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

