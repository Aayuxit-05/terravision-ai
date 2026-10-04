# 🌍 TerraVision AI

**TerraVision AI** is a deep learning-based satellite image classification project that helps identify different types of land and landscapes from satellite imagery.

It uses **MobileNetV2 transfer learning** to classify satellite images into 10 land-use categories.

## 🛰️ What Can It Identify?

TerraVision AI can classify images into:

* 🌾 Annual Crop
* 🌲 Forest
* 🌿 Herbaceous Vegetation
* 🛣️ Highway
* 🏭 Industrial
* 🌱 Pasture
* 🌾 Permanent Crop
* 🏠 Residential
* 🌊 River
* 🌅 Sea/Lake

## 🧠 How It Works

**Satellite Image → Deep Learning Model → Land-Use Prediction → Confidence Score**

The project uses **MobileNetV2**, a pre-trained convolutional neural network, with transfer learning.

The model was trained for 3 epochs using the **EuroSAT RGB dataset**.

## 📊 Model Performance

The model achieved approximately **92% validation accuracy** on the EuroSAT validation data.

Performance can vary between different land-use classes, with some visually similar categories such as rivers, highways, and vegetation being more challenging to distinguish.

## 📦 Dataset

The project uses the **EuroSAT RGB dataset**, containing 27,000 labeled satellite image patches across 10 land-use classes.

Dataset source:
https://zenodo.org/records/7711810

## 🚀 Running the App

Install the required Python libraries and run:

```bash
py -m streamlit run app.py
```

Then upload a satellite image through the web interface to receive a predicted land-use class and confidence score.

## 🌍 Possible Applications

TerraVision AI could be expanded for:

* 🌲 Forest monitoring
* 🌊 Water-body monitoring
* 🌾 Agricultural analysis
* 🏙️ Urban expansion monitoring
* 🌪️ Disaster and environmental assessment
* 🗺️ Large-scale land-use mapping

## ⚠️ Limitations

The current model is a student-level prototype trained on EuroSAT RGB images. It works best with satellite imagery that is visually similar to the training dataset.

The confidence score represents the model's prediction strength and does not guarantee that the prediction is correct.

## 🔮 Future Scope

Future versions could include:

* More satellite datasets
* Higher-resolution imagery
* Multispectral satellite data
* Region highlighting
* Change detection over time
* Automated environmental reports
* Large-scale satellite image analysis

## 👩‍💻 Project

**TerraVision AI**
First-year B.Tech AI/ML project
Built using **Python, TensorFlow, MobileNetV2 and Streamlit**.
