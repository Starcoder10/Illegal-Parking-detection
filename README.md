llegal Parking Detection System
Automated detection of illegally parked vehicles using Digital Image Processing and Deep Learning

PythonOpenCVYOLOv8Tkinter

📋 Overview
A real-time system that detects vehicles parked in no-parking zones using CCTV video footage. Built with a 10-stage image processing pipeline combining deep learning (YOLOv8) with classical DIP techniques (CLAHE, MOG2, morphological operations).

✨ Key Features
🎯 YOLOv8 Vehicle Detection — Real-time detection of cars, motorcycles, buses, and trucks
🌙 Night Mode (CLAHE) — Adaptive histogram equalization for low-light video enhancement
🔄 Hybrid IoU + Centroid Tracking — Persistent vehicle ID tracking across frames
⏱️ Smart Stationarity Detection — Centroid stability analysis distinguishes temporary stops from illegal parking (reduces false positives by ~83%)
🖼️ MOG2 Background Subtraction — Shadow and noise removal using morphological operations
📸 Automatic Evidence Capture — Timestamped violation screenshots saved as proof
🔥 Violation Heatmap — Visual overlay showing parking violation hotspots
📊 Analytics Dashboard — Real-time charts, statistics, and violation log
💾 CSV Export — Exportable violation reports for traffic authorities
🏗️ System Architecture
Video Input → CLAHE Preprocessing → YOLOv8 Detection → IoU+Centroid Tracking
                    ↓                                           ↓
            MOG2 Background                              ROI Zone Check
            Subtraction                                        ↓
                                                     Centroid Stability
                                                        Analysis
                                                           ↓
                                                    Parking Timer
                                                           ↓
                                              Violation → Evidence Capture
                                                           ↓
                                                   Dashboard + Reports
📁 Project Structure
├── main.py                    # Entry point — 10-step detection pipeline
├── dashboard.py               # Tkinter GUI — dark theme dashboard
├── detector.py                # YOLOv8 vehicle detection
├── tracker.py                 # Hybrid IoU + Centroid multi-object tracker
├── roi.py                     # No-parking zone definition & point-in-polygon
├── timer_check.py             # Dwell-time management with stationarity gating
├── visualizer.py              # Frame annotation, HUD, heatmap rendering
├── preprocessor.py            # CLAHE low-light enhancement
├── background_subtractor.py   # MOG2 + morphological post-processing
├── evidence.py                # Violation screenshot capture
├── train.py                   # Custom model training on Roboflow dataset
└── requirements.txt           # Dependencies
🚀 Quick Start
1. Install Dependencies
bash
pip install -r requirements.txt
2. Run the Application
bash
python main.py
3. Usage
Upload a CCTV/traffic video (MP4, AVI, MOV)
Define no-parking zones by clicking on the video frame
Configure time threshold, detection confidence, night mode
Start Detection and monitor violations in real-time
Export CSV report or browse evidence screenshots
🔧 Technologies Used
Technology	Purpose
YOLOv8	Real-time vehicle detection (COCO pre-trained)
OpenCV	Image processing, video I/O, morphological operations
CLAHE	Low-light frame enhancement on LAB color space
MOG2	Background subtraction for motion/stationarity analysis
NumPy	Numerical computation and array operations
Matplotlib	Embedded analytics charts
Tkinter	Desktop GUI with dark Catppuccin theme
Roboflow	Dataset management (Parking-AMU50)
📊 Performance
Metric	Result
Processing Speed (CPU)	8–12 FPS
Processing Speed (GPU)	20–30 FPS
False Positive Reduction	~83% (with stability gating)
Night Detection Improvement	+20–25% (with CLAHE)
Stationarity Detection Latency	~0.2 seconds
🧠 DIP Techniques Applied
CLAHE — Contrast enhancement in LAB color space
MOG2 Background Subtraction — Foreground/background separation
Morphological Erosion & Dilation — Noise and shadow removal
Gaussian Blur — Frame denoising
Point-in-Polygon (Ray Casting) — Zone membership testing
IoU Computation — Bounding box overlap matching
Centroid Euclidean Distance — Object association
Heatmap Generation — Violation density visualization
📄 Dataset
Parking-AMU50 — Illegal Parking Dataset

Source: Roboflow Universe
Format: YOLOv8
🏷️ Domain
Smart Transportation · Smart Cities · Digital Image Processing

📝 License
This project is developed as part of an academic Digital Image Processing course.
