import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Title
st.set_page_config(page_title="Vehicle Detection & Counting - YOLOv8", layout="wide")
st.title("🚗 Vehicle Detection & Counting (YOLOv8)")
st.write("Upload a video or use webcam to detect and count vehicles in real-time.")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  

model = load_model()

# Sidebar options
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Choose input
source = st.radio("Select Input Source:", ["Upload Video", "Webcam"])

# -------------------------------------------------------------------------
# CORE LOGIC: TRACKING & COUNTING
# -------------------------------------------------------------------------
def process_frame(frame, unique_ids):
    # 1. Use TRACKING (persist=True) to assign IDs to cars
    results = model.track(frame, conf=conf_threshold, persist=True)
    annotated_frame = results[0].plot()
    
    # Target Classes: Car(2), Motorcycle(3), Bus(5), Truck(7)
    target_classes = [2, 3, 5, 7] 
    
    # 2. Extract IDs if detections exist
    if results[0].boxes.id is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for cls, track_id in zip(clss, track_ids):
            if int(cls) in target_classes:
                unique_ids.add(track_id) # Add ID to the set (sets only keep unique values)

    # 3. Calculate Total Count
    current_count = len(unique_ids)

    # 4. Draw Counter on Frame (Top-Right)
    text = f"Total: {current_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5 
    font_thickness = 3
    text_color = (0, 0, 255) # Red (BGR)

    # Calculate Text Size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    img_height, img_width = annotated_frame.shape[:2]
    x_pos = img_width - text_width - 20
    y_pos = 50
    
    # Draw White Background for readability
    cv2.rectangle(annotated_frame, (x_pos - 10, y_pos - text_height - 10), 
                  (x_pos + text_width + 10, y_pos + 10), (255, 255, 255), -1)
    
    # Draw Text
    cv2.putText(annotated_frame, text, (x_pos, y_pos), font, font_scale, text_color, font_thickness)

    return annotated_frame, current_count
# -------------------------------------------------------------------------

if source == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file:
        # Save uploaded file to temp with .mp4 extension
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close() 
        
        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("Error: Could not open video.")
        else:
            # Prepare Output Video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            output_temp.close()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out_writer = cv2.VideoWriter(output_temp.name, fourcc, fps, (width, height))
            
            stframe = st.empty()
            count_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0
            
            # *** Initialize Unique ID Set for this video ***
            video_unique_ids = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process Frame (Pass the set!)
                annotated_frame_bgr, count = process_frame(frame, video_unique_ids)

                # Convert to RGB for Streamlit Display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

                # Update Streamlit UI
                stframe.image(annotated_frame_rgb, channels="RGB")
                count_placeholder.markdown(f"### 🚦 Total Vehicles Detected: **{count}**")

                # Write BGR frame to Output Video
                out_writer.write(annotated_frame_bgr)
                
                frame_idx += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            out_writer.release()
            os.remove(tfile.name) 

            st.success("Processing complete!")
            
            with open(output_temp.name, 'rb') as f:
                video_bytes = f.read()

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="processed_output.mp4",
                mime="video/mp4"
            )
            os.remove(output_temp.name)

elif source == "Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        stframe = st.empty()
        count_placeholder = st.empty()
        
        # *** Initialize Unique ID Set for webcam ***
        webcam_unique_ids = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame_bgr, count = process_frame(frame, webcam_unique_ids)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)

            stframe.image(annotated_frame_rgb, channels="RGB")
            count_placeholder.markdown(f"### 🚦 Total Vehicles Detected: **{count}**")

        cap.release()