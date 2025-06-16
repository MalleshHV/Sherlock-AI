
# 🚨 Sherlock-AI: Revolutionizing Crime Investigations with Artificial Intelligence

## 🧠 Overview
**Sherlock-AI** is an AI-powered platform designed to enhance public safety and support crime investigation through real-time weapon detection and a crime analysis dashboard. It combines:
- **YOLOv8** for computer vision-based weapon detection
- **Streamlit** dashboard for crime data analysis, legal research, and news aggregation

## 🚀 Key Features
- 🔫 **Real-Time Weapon Detection**  
  Live object detection from webcam/video using YOLOv8, with sound, email alerts, and logs.
- 🧩 **Crime Investigation Assistant**  
  Chat-based system with RAG and Transformers to analyze crimes and suggest insights.
- 📚 **Legal Document Analyzer**  
  Extracts text from legal PDFs and answers questions based on extracted content.
- 📊 **Interactive Crime Dashboard**  
  Dynamic charts, maps (Plotly, Folium) for visualizing crime data.
- 📰 **Real-Time Crime News**  
  Fetch and analyze news from News API, filtered by country/region.
- 🛡️ **Crime Prevention Insights**  
  Offers AI-generated suggestions for public safety and legal protection.
- 🌃 **Image Enhancement**  
  CLAHE-enhanced frames for better vision performance in low-light scenes.
- ⚡ **GPU Optimization**  
  Supports CUDA-based acceleration for faster processing of models.

---

## 🧩 Prerequisites

- Python 3.8+
- Required packages:
  - `opencv-python`, `ultralytics`, `streamlit`, `transformers`, `torch`, `bitsandbytes`
  - `plotly`, `folium`, `pandas`
- **News API Key** (https://newsapi.org)
- **Ngrok** account + auth token
- Webcam or test images
- YOLOv8 model weights (e.g., `yolov8x.pt`)

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sherlock-ai.git
cd sherlock-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File
```env
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENT=recipient_email@gmail.com
WEAPON_CONFIDENCE=0.25
PERSON_CONFIDENCE=0.3
NOTIFICATION_COOLDOWN=60
MODEL_PATH=WDS/yolov8x.pt
NEWS_API_KEY=your_news_api_key
```

> 🔐 **Important:** Use Gmail App Passwords or OAuth tokens instead of storing plain passwords.

### 4. Set Up Ngrok
```bash
ngrok config add-authtoken your_ngrok_auth_token
```

---

## 🧠 Model Setup

### Automatic Model Download
When `main.py` is executed, the YOLOv8 model (e.g., `yolov8x.pt`) will be automatically downloaded by the `ultralytics` library if it’s missing.

### Manual Model Download (Optional)
1. Visit [Ultralytics YOLOv8 Releases](https://github.com/ultralytics/ultralytics/releases).
2. Download `yolov8x.pt` (or smaller variants like `yolov8n.pt`).
3. Place it in the `WDS/` folder.
4. Confirm `MODEL_PATH=WDS/yolov8x.pt` in your `.env`.

---

## 🛡️ Usage

### 1. Weapon Detection System (WDS)
```bash
cd WDS
python main.py
```
This will:
- Load the YOLOv8 model
- Start webcam monitoring
- Detect weapons with alerting and logging
- Press `q` to stop monitoring

---

### 2. Crime Investigation Dashboard
Start the dashboard:
```bash
streamlit run app3.py
```

To make it public via Ngrok:
```python
from pyngrok import ngrok
public_url = ngrok.connect(8501, "http")
print("Access the dashboard at:", public_url)
```

---

## 📂 Project Structure
```
sherlock-ai/
│
├── check.py                   # Streamlit dashboard                      # Configuration file (not tracked)
├── requirements.txt
│
├── WDS/                      # Weapon Detection System
│   ├── main.py               # Live monitoring script
│   ├── .env 
│   ├── yolov8x.pt            # YOLO model (downloaded separately)
│   ├── detection_logs/       # Weapon detection logs
│   └── test_images/          # Sample images for testing
```

---

## 🔧 Configuration Options

| Variable             | Description                                   | Default         |
|----------------------|-----------------------------------------------|-----------------|
| `WEAPON_CONFIDENCE`  | Minimum confidence for weapon detection       | `0.25`          |
| `PERSON_CONFIDENCE`  | Confidence threshold for detecting people     | `0.30`          |
| `NOTIFICATION_COOLDOWN` | Time between alerts (in seconds)          | `60`            |
| `MODEL_PATH`         | Path to YOLOv8 model weights                  | `WDS/yolov8x.pt`|
| `NEWS_API_KEY`       | API key for fetching crime news               | *Required*      |

---

## 🧑‍💻 Contributing

1. Fork the repo
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
