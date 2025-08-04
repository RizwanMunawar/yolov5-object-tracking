# yolov5-object-tracking

<a href="https://deepwiki.com/RizwanMunawar/yolov5-object-tracking"><img src="https://img.shields.io/badge/Repo-DeepWiki-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="YOLOv5-object-tracking DeepWiki"></a>

### New Features
- YOLOv5 Object Tracking Using Sort Tracker
- Added Object blurring Option
- Added Support of Streamlit Dashboard
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

### Pre-Requsities
- Python 3.9 (Python 3.7/3.8 can work in some cases)

### Steps to run Code
1 - Clone the repository
```
git clone https://github.com/RizwanMunawar/yolov5-object-tracking.git
```

2 - Goto the cloned folder.
```
cd yolov5-object-tracking
```

3 - Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
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

4 - Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

5 - Install requirements with mentioned command below.
```
pip install -r requirements.txt
```

6 - Run the code with mentioned command below.
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

7 - Output file will be created in the working-dir/runs/detect/exp with original filename

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
1. [YOLOv7 Training on Custom Data](https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623) – Guide to training YOLOv7 on custom datasets.
2. [Roadmap for Computer Vision Engineer](https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c) – A step-by-step career guide for aspiring computer vision engineers.
3. [YOLOR or YOLOv5: Which One is Better?](https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1) – Comparative analysis of YOLOR vs. YOLOv5 for model selection.
4. [Train YOLOR on Custom Data](https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6) – Instructions for customizing YOLOR on unique datasets.
5. [Develop an Analytics Dashboard Using Streamlit](https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f) – Tutorial on building data dashboards with Streamlit.
6. [Jetson Nano in Computer Vision Solutions](https://medium.com/augmented-startups/jetson-nano-is-rapidly-involving-in-computer-vision-solutions-5f588cb7c0db) – Insight on Jetson Nano's role in embedded AI projects.
7. [How Computer Vision Products Help in Warehouses](https://chr043416.medium.com/how-can-computer-vision-products-help-in-warehouses-aa1dd95ec79c) – Overview of computer vision applications in warehouse efficiency.

For more details, you can reach out to me on [Medium](https://chr043416.medium.com/) or can connect with me on [LinkedIn](https://www.linkedin.com/in/muhammadrizwanmunawar/)
