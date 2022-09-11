# yolov5-object-tracking

### New Features
- YOLOv5 Object Tracking Using Sort Tracker
- Added Object blurring Option
- Added Support of Streamlit Dashboard
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Option to crop and save detected objects

### Steps to run Code
- Clone the repository
```
git clone https://github.com/RizwanMunawar/yolov5-object-tracking.git
```

- Goto the cloned folder.
```
cd yolov5-object-tracking
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users
python3 -m venv yolov5objtracking
source yolov5objtracking/bin/activate

### For Window Users
python3 -m venv yolov5objtracking
cd yolov5objtracking
cd Scripts
activate
cd ..
cd ..
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```

- Run the code with mentioned command below.
```
#for detection only
python ob_detect.py --weights yolov5s.pt --source "your video.mp4"

#for detection of specific class (person)
python ob_detect.py --weights yolov5s.pt --source "your video.mp4" --classes 0

#for object detection + object tracking
python obj_det_and_trk.py --weights yolov5s.pt --source "your video.mp4"

#for object detection + object tracking + object blurring
python obj_det_and_trk.py --weights yolov5s.pt --source "your video.mp4" --blur-obj

#for object detection + object tracking of specific class (person)
python obj_det_and_trk.py --weights yolov5s.pt --source "your video.mp4" --classes 0
```

- Output file will be created in the working-dir/runs/detect/exp with original filename

### Streamlit Dashboard
- If you want to run detection on streamlit app (Dashboard), you can use mentioned command below.
```
python -m streamlit run app.py
```

