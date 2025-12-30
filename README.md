# Clash Royale Emote Controller

This project uses **AI-powered Computer Vision** to map your facial expressions and hand gestures to iconic Clash Royale emotes in real-time. By leveraging **MediaPipe** and **OpenCV**, the application detects specific "trigger" gestures to overlay emotes directly onto the video feed.

## The Goal

The primary objective of this project was to experiment with:

* **Face Mapping**: Tracking 468 facial landmarks to detect smile ratios and proximity.
* **Hand Tracking**: Detecting complex gestures like closed fists and spatial proximity to facial features.
* **Real-time Interaction**: Creating a low-latency "Digital Puppet" experience where the user's movements control the visual output.

## 🛠 Features

* **Dual Window Output**:
1. **Output Window**: A clean view showing only the triggered Clash Royale emotes.
2. **Input Tracker**: A secondary window showing the raw face mesh and hand skeletal tracking for debugging.


* **Gesture Mapping**:
* **Smile**: Triggers `smile.webp` using a width-to-eye ratio.
* **Crying (Double Fists)**: Triggers `boohoo.webp` when both fists are held near the mouth.
* **Hide (Open Hands)**: Triggers `hide.webp` when both hands are open and covering the face.
* **Neutral**: Displays `neutral.webp` when no specific gestures are detected.



## 📂 Project Structure

```text
Clash-Emotes/
├── Main.py            # Main application logic
├── README.md          # Project documentation
├── Assets/            # Folder containing .webp emote files
│   ├── smile.webp
│   ├── boohoo.webp
│   ├── hide.webp
│   └── neutral.webp
└── .venv/             # Python Virtual Environment

```

## ⚙️ How to Run

To set up this project on your machine, follow these steps:

1. **Create and Activate Environment**:
```bash
python3 -m venv .venv
source .venv/bin/python

```


2. **Install Dependencies**:
```bash
python3 -m pip install opencv-python mediapipe

```


3. **Run the Application**:
Right-click `Main.py` in PyCharm and select **Run**, or use the terminal:
```bash
python3 Main.py

```



## 📝 Technical Details

1. **Landmark Extraction**: The system identifies 3D face landmarks and 21 hand landmarks per hand.
2. **Normalized Coordinates**: Gesture detection uses normalized  and  coordinates ( to ) to ensure the system works regardless of distance from the camera.
3. **Proximity Bubbles**: Hand triggers are calculated using Euclidean distance between hand knuckles and the center of the face.
