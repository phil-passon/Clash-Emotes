# Real-Time Gesture-Controlled AR Emote Overlay

An advanced computer vision application using **MediaPipe** and **OpenCV** to map human facial expressions and hand gestures to a priority-based Augmented Reality (AR) overlay system. The project demonstrates high-fidelity tracking and complex gesture state management.

## Technical Architecture

The system utilizes two primary MediaPipe solutions running concurrently:

1. **Face Mesh (Refined)**: Tracking 468 3D landmarks, including iris and lip-contour refinement for high-precision smile-ratio calculation.
2. **Hand Landmarks**: Simultaneous tracking of two hands with 21 3D landmarks each, enabling complex pose estimation (e.g., finger state detection, fist clenching, and spatial proximity).

### Gesture Logic & Priority Matrix

To prevent frame flickering and gesture overlap, the system employs a hierarchical priority check. If multiple conditions are met, the higher-priority emote is rendered.

| Priority | Emote Trigger | Logic Implementation |
| --- | --- | --- |
| **1** | **Hide** | 2-Hand Open Palm detection + Nose Proximity ( units). |
| **2** | **Waiting** | 1-Hand Logic: Palm at chin level + Index tip  Eye level. |
| **3** | **Peace** | Finger state: Index/Middle Extended + Ring/Pinky Folded. |
| **4** | **Flex** | 2-Fist detection + Knuckle  Eye level + Face Margin . |
| **5** | **Excited** | Smile Ratio  + Hands at Mouth level (Fingers below nose). |
| **6** | **Boohoo** | 2-Fist detection + Mouth Proximity ( units). |
| **7** | **Smile** | Lip Landmark Distance Ratio . |
| **8** | **Neutral** | Default state; no active flags detected. |

---

## Installation & Deployment

### Environment Setup

* **Python**: 3.8 or higher.
* **Dependencies**:
```bash
pip install opencv-python mediapipe

```



### Asset Requirements

The script expects a directory named `Assets/` in the root folder containing the following standardized image files:
`neutral.png`, `smile.png`, `excited.png`, `boohoo.png`, `flex.png`, `hide.png`, `peace.png`, `waiting.png`.

### Execution

```python
python main.py

```

---

## Technical Features

* **Coordinate Normalization**: Landmarks are calculated in normalized coordinates ( to ), ensuring the triggers work regardless of camera resolution or aspect ratio.
* **Vertical Zone Segmentation**: Uses the -coordinates of the Nose Tip (Landmark 1) and Eyes (Landmark 33) as dynamic horizontal planes to separate "Excited" (lower zone) from "Waiting" (upper zone).
* **Euclidean Proximity Checks**: Triggers are based on the mathematical distance between hand nodes and facial nodes ().

## Repository Structure

```text
├── Assets/             # Emote image overlays
├── main.py             # Main execution script
└── README.md           # Technical documentation

```
