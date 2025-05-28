
# 🚜 Agriculture CNN

This project uses a YOLOv5-based Convolutional Neural Network to detect agricultural objects from images and video feeds. It currently supports detection of **8 different agricultural item classes**, trained on real-world field data.

---

## 🎯 Goal

To develop and test a robust, scalable object detection model that identifies **tractors, trailers, animals, and tools** commonly found in agricultural settings, using deep learning.

---

## 🧪 Technologies Used

- 🐍 Python 3.x  
- 🔭 YOLOv5 (Ultralytics)  
- 🧪 OpenCV  
- 📈 Matplotlib & Pandas (for analysis/visualization)  
- 🛠 os, shutil, subprocess, time  
- 🐙 Git for version control  

---

## 🗂️ Project Structure

```
Agriculture-CNN/
├── dataset1/ ... dataset21/   # Original YOLO datasets (images & labels)
├── merged/                    # Final merged dataset
├── test_tractors/            # Test set
├── yolov5/                   # Cloned YOLOv5 repository
├── .gitignore                # Ignored files
├── data.yaml                 # Dataset config (classes, paths)
├── futtato.py                # Run detection on image folder
├── merger.py                 # Merge datasets into one
├── teacher.py                # Train the YOLOv5 model
├── video.py                  # Run detection on live video feed
```

---

## ✅ Features

- 🔍 Merges 21 separate datasets into one training set  
- 🏷️ Trained YOLOv5 model with custom agricultural labels  
- 📸 Real-time detection from webcam/video using `video.py`  
- 🖼️ Batch image folder processing using `futtato.py`  
- 📊 Visualization utilities using matplotlib and pandas  

---

## 🧠 Current Labels (8/27)

These are defined in `data.yaml`:
```yaml
names:
  - traktor #elég decens
  - kabin
  - utanfuto #megvan
  - balazo
  - vetogep
  - permetezo
  - bala #megvan
  - markolo_kar #megvan
  - krumpliszedo
  - szemelyek #megvan
  - foliazo
  - eke 
  - szenagyujto
  - tarcsa
  - tartalypotkocsi
  - krumplivetogep
  - mutragyaszoro
  - viztartaly # na meg ez is
  - kapagep
  - csomagolt_bala
  - gumiabroncs #megvan
  - kaszagep
  - funyiro_traktor
  - takarmany_kevero
  - szarvasmarha # ez is
  - kombajn
  - kombajn_asztal
```

---

## 🚀 How to Run

### 1. Clone YOLOv5 (if not already)
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Train the model
```bash
python3 teacher.py
```

### 3. Run on test images
```bash
python3 futtato.py
```

### 4. Run live detection from camera
```bash
python3 video.py
```

---

## 📦 Requirements

Install using:
```bash
pip install -r yolov5/requirements.txt
pip install matplotlib pandas opencv-python
```

---

## 🔄 Dataset Merge

You can regenerate the merged dataset using:
```bash
python3 merger.py
```
This script combines all `dataset*/images` and `dataset*/labels` into the `merged/` folder.

---

## 🧪 Evaluation

- 📈 Total images: **494**  
- 🏷️ Total labels: **461**  
- 👁️ Model tested on both still images and real-time webcam input  

---

## 🔜 Planned Features

- [ ] Increase label diversity (more tools/animals)  
