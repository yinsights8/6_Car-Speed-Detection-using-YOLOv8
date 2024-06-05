import cv2
import cvzone
import math
from utils.sort import *
from utils.ImgResizer import ImageResizer
from ultralytics import YOLO

###################################################################################################################################################

# Video Path
# E:/Data Science/Ineuron/Notebooks/4. Computer Vision/8. Object Detection/3_Yolov5_Custom_Training-main/Complete YOLO/Videos/object_track/test_sample.mp4
Video_Path = "E:/My_Model/3._Object_Detection/6_Car Speed Detector/Videos/highway.mp4"

# Weights
weights = "E:/My_Model/3._Object_Detection/6_Car Speed Detector/Weights/yolov8m.pt"

# coco dataset classes
# this model is already trained on coco dataset
class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "firehy drant",
    "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eyeglasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
    "bottle", "plate", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "mirror", "diningtable", "window",
    "desk", "toilet", "door", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddybear", "hairdrier",
    "toothbrush", "hairbrush"
]

###################################################################################################################################################



cam = cv2.VideoCapture(Video_Path)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

model = YOLO(weights)

myColor = (255, 0, 255)

L1_limits = [191, 248, 924, 248]
L2_limits = [0, 346, 1135, 346]

red_line_y = 248
blue_line_y = 346
offset = 7

Dwn = dict()
Up = dict()

CountUp = []
CountDwn = []

while True:
    success, img = cam.read()
    image = ImageResizer().resize(image=img)
    results = model(image, True)

    detections = np.empty((0, 5))

    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100  # get the confidence from box
            Classes = int(box.cls[0])
            currentClass = class_names[Classes]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                CurrentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((CurrentArray, detections))

        # update the tracker
    resutlTracker = tracker.update(dets=detections)

    cv2.line(img=image, pt1=(L1_limits[0], L1_limits[1]), pt2=(L1_limits[2], L1_limits[3]), color=(0, 0, 255), thickness=1)  # L1
    cv2.line(img=image, pt1=(L2_limits[0], L2_limits[1]), pt2=(L2_limits[2], L2_limits[3]), color=(255, 0, 0), thickness=1)  # L2

    for result in resutlTracker:
        x1, y1, x2, y2, id = result
        w, h = x2 - x1, y2 - y1
        x1, y1, w, h, id = int(x1), int(y1), int(w), int(h), int(id)
        cx, cy = x1 + w // 2, y1 + h // 2

        # When cars going from red line to blue line
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            Dwn[id] = time.time()
        if id in Dwn:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsedTime1 = time.time() - Dwn[id]
                if CountDwn.count(id) == 0:
                    CountDwn.append(id)
                    distance = 10
                    speedMs = distance / elapsedTime1
                    SpeedKmph = round(speedMs * 3.9, 2)

                    Text = f"ID: {id} Speed: {SpeedKmph} Kmh"
                    cv2.circle(image, (cx, cy), 4, (50, 50, 255), cv2.FILLED)
                    cv2.line(img=image, pt1=(L2_limits[0], L2_limits[1]), pt2=(L2_limits[2], L2_limits[3]), color=(0, 255, 0),
                             thickness=2)
                    cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.5, thickness=1,
                                       font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=myColor)

        # When cars going from Blue line to Red line
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            Up[id] = time.time()
        if id in Up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsedTime2 = time.time() - Up[id]
                if CountUp.count(id) == 0:
                    CountUp.append(id)
                    distance1 = 10
                    speedMs = distance1 / elapsedTime2
                    SpeedKmph = round(speedMs * 3.9, 2)

                    Text = f"ID: {id} Speed: {SpeedKmph} Kmh"
                    cv2.circle(image, (cx, cy), 4, (50, 50, 255), cv2.FILLED)
                    cv2.line(img=image, pt1=(L2_limits[0], L2_limits[1]), pt2=(L2_limits[2], L2_limits[3]), color=(0, 255, 0),
                             thickness=2)
                    cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.5, thickness=1,
                                       font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=myColor)

    cvzone.putTextRect(image, text=f"CountDown = {len(CountDwn)}", pos=(40, 30), scale=0.5,
                       thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(255, 0, 255))
    cvzone.putTextRect(image, text=f"CountUp = {len(CountUp)}", pos=(40, 50), scale=0.5,
                       thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(255, 0, 255))

    cv2.imshow("Image", image)
    if cv2.waitKey(10) & 0xFf == ord('q'):
        break
