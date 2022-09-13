# yolov5-object-tracking

### New Features
- YOLOv5 Object Tracking Using Sort Tracker
- Added Object blurring Option
- Added Support of Streamlit Dashboard
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

### Coming Soon
- Option to crop and save detected objects
- Dashboard design enhancement

### Pre-Requsities
- Python 3.9 (Python 3.7/3.8 can work in some cases)

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

#for object detection + object tracking + object blurring + different color for every bounding box
python obj_det_and_trk.py --weights yolov5s.pt --source "your video.mp4" --blur-obj --color-box

#for object detection + object tracking of specific class (person)
python obj_det_and_trk.py --weights yolov5s.pt --source "your video.mp4" --classes 0
```

- Output file will be created in the working-dir/runs/detect/exp with original filename

### Streamlit Dashboard
- If you want to run detection on streamlit app (Dashboard), you can use mentioned command below.

<b>Note:</b> Make sure, to add video in the <b>yolov5-object-tracking</b> folder, that you want to run on streamlit dashboard. Otherwise streamlit server will through an error.
```
python -m streamlit run app.py
```

<table>
  <tr>
    <td>YOLOv5 Object Detection</td>
    <td>YOLOv5 Object Tracking</td>
    <td>YOLOv5 Object Tracking + Object Blurring</td>
    <td>YOLOv5 Streamlit Dashboard</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/189525324-9aaf4b60-9336-41c3-8a27-8722bb7da731.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525332-1e84b4d5-ae4e-4c1b-9498-0ec1d4ad4bd7.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525328-f85ef474-e964-4d79-8f75-78ad4e5397d4.png"></td>
     <td><img src="https://user-images.githubusercontent.com/62513924/189525342-8d4d81f4-5e3a-45aa-9972-5f5de1c72159.png"></td>
  </tr>
 </table>

### References
 - https://github.com/ultralytics/yolov5
 - https://github.com/abewley/sort
 
### My Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
- https://medium.com/augmented-startups/jetson-nano-is-rapidly-involving-in-computer-vision-solutions-5f588cb7c0db
- https://chr043416.medium.com/how-can-computer-vision-products-help-in-warehouses-aa1dd95ec79c

For more details, you can reach out to me on [Medium](https://chr043416.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)
