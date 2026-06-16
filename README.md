# 🌱 Agri Prognosis: Intelligent Plant Health Prediction System

## 📖 Overview

Agri Prognosis is a Smart Agriculture System that combines **Internet of Things (IoT)**, **Machine Learning (ML)**, **Cloud Computing**
and **Web/Mobile Technologies** to monitor soil conditions and predict plant health.

The system collects important soil parameters such as:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Soil pH
* Soil Moisture
* Plant Type

Using these parameters, a Machine Learning model predicts whether a plant is **Healthy** or **Unhealthy** and provides suitable recommendations.

---

## 🎯 Problem Statement

Traditional farming methods often rely on manual soil testing and visual crop inspection, which can be time-consuming and may not provide timely 
information about soil conditions.

Agri Prognosis addresses this challenge by:

* Monitoring soil conditions
* Predicting plant health
* Providing decision support
* Improving resource utilization
* Supporting smart farming practices

---

## 🚀 Key Features

✅ Real-time soil nutrient monitoring (NPK)

✅ Soil pH measurement

✅ Soil moisture monitoring using IoT

✅ ThingSpeak cloud integration

✅ Plant health prediction using Machine Learning

✅ Flask-based Web Application

✅ Android Mobile Application

✅ Multilingual Support

✅ Cloud Deployment using Render

---

## 🛠️ Technologies Used

### Hardware

* Raspberry Pi
* NPK Sensor
* Soil Moisture Sensor
* ADC Module
* NodeMCU ESP8266
* 3-in-1 Soil pH Meter

### Software

* Python
* Flask
* HTML
* CSS
* JavaScript
* Bootstrap
* Android Studio
* ThingSpeak
* Render
* GitHub

### Machine Learning

* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* XGBoost

---

## ⚙️ System Workflow

1. Soil parameters are collected using sensors.
2. Raspberry Pi processes sensor readings using NPK Sensor.
3. pH values are taken using 3-in-1 pH meter.
4. Moisture data is uploaded to ThingSpeak Cloud.
5. Agricultural data is preprocessed.
6. Machine Learning models are trained and evaluated.
7. SVM is selected as the final prediction model.
8. User enters soil parameters through the application.
9. Flask backend performs prediction.
10. Plant health status and recommendations are displayed.

---

## 📊 Machine Learning Performance

The following models were evaluated:

| Model         | Accuracy |
| ------------- | -------- |
| Decision Tree | ~83%     |
| Random Forest | ~83%     |
| XGBoost       | ~83%     |
| SVM           | ~90–93%  |

**Best Model:** Support Vector Machine (SVM)

---

## 📁 Project Structure

```text
agri-prognosis-app/
│
├── app.py
├── health_model.pkl
├── scaler.pkl
├── templates/
├── static/
├── dataset/
├── Android_App/
├── requirements.txt
└── README.md
```

---

## ▶️ Installation

### Clone Repository

```bash
git clone https://github.com/ThotapallySindhuja/agri-prognosis-app.git
```

### Navigate to Project Folder

```bash
cd agri-prognosis-app
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📱 Application Interfaces

### Web Application

Users can enter soil parameters and receive plant health predictions through a browser-based interface.

### Android Application

The Android application provides mobile access to the prediction system using WebView integration.

---

## ☁️ Cloud Integration

ThingSpeak is used for:

* Soil moisture monitoring
* Data visualization
* Historical data storage
* Remote access

Render is used for:

* Flask application hosting
* Public deployment
* Web accessibility

---

## 🔮 Future Enhancements

* Profitability Analysis
* Next Crop Cycle Recommendation
* Fertilizer Recommendation System
* Support for Multiple Crop Varieties
* Solar-Powered Monitoring Infrastructure

---

## 📜 License

This project is developed for academic and research purposes.
