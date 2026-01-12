# 🚗 Vehicle Detection & Counting (YOLOv8)

This computer vision project detects and counts vehicles from traffic videos in real time using **YOLOv8** and **OpenCV**.  
It identifies cars, motorcycles, buses, and trucks, displays the count per frame, and saves an annotated output video.  
This system can be used for traffic monitoring, smart city projects, and transportation research.

---

## 🎥 Output Video

### 🔸 Vehicle Detection & Counting in Action  
<!-- <video src="https://github.com/amadokjo/vehicle-detection-streamlit/blob/main/output.mp4" width="400" controls></video> -->

<video src="https://github.com/amadokjo/vehicle-detection-streamlit/blob/main/output.gif" width="400" controls></video>

---

## 🔴 Live Demo

---

## ⚙️ How It Works

1. Load the **YOLOv8 pre-trained model** (`yolov8n.pt` for speed or `yolov8s.pt` for higher accuracy)
2. Process video frames using **OpenCV**
3. Run YOLOv8 object detection on each frame
4. Filter detections for **vehicle classes**: car, motorcycle, bus, truck
5. Draw **bounding boxes** and labels with confidence scores
6. Count vehicles in the current frame
7. Overlay the vehicle count onto the frame
8. Save the processed frames into an **output video file**

---

## 🧠 Technologies Used

- Python 3
- YOLOv8 (Ultralytics)
- OpenCV

---

## 📦 Requirements

```bash
ultralytics
opencv-python
```

## 📁 Project Structure

| File                          | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `vehicle_detection_yolov8.py` | Main script that detects and counts vehicles in the video |
| `video.mp4`                   | Input traffic video                                       |
| `output.mp4`                  | Output video with detected vehicles and counts            |
| `requirements.txt`            | Python libraries used                                     |
| `README.md`                   | Project documentation                                     |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python vehicle_detection_yolov8.py
```

---

🔭 Future Enhancements

Lane-wise counting using virtual lines/regions of interest (ROI)

Speed estimation using pixel-to-meter calibration and FPS

Real-time CCTV/IP camera stream integration

Dashboard/analytics for live traffic monitoring

Train YOLOv8 on a custom dataset for improved class coverage (e.g., rickshaw, bicycle)

---

## 👨‍💻 Author


---

⭐ If you found this project useful, please consider starring the repository!
