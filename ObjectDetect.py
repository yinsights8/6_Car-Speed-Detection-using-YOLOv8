import cv2
import cvzone
import math
from utils.sort import *
from ultralytics import YOLO

###################################################################################################################################################

# Video Path
# E:/Data Science/Ineuron/Notebooks/4. Computer Vision/8. Object Detection/3_Yolov5_Custom_Training-main/Complete YOLO/Videos/object_track/test_sample.mp4
Video_Path = "E:/My_Model/3._Object_Detection/6_Car Speed Detector/Videos/2.mp4"
# Video_Path = ("E:/Data Science/Ineuron/Notebooks/4. Computer Vision/8. Object Detection/3_Yolov5_Custom_Training-main/Complete "
#               "YOLO/Videos/object_track/test_sample.mp4")

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
def resize_Video(Image, scale_percentage=50):
    # Get the current frame size
    height, width, _ = Image.shape
    print(height, width)
    if height == 480 and width == 640:
        resized = Image
    else:
        # Resize the frame
        scale_percent = scale_percentage
        new_width = int(width * scale_percent / 100)
        new_height = int(height * scale_percent / 100)
        dim = (new_width, new_height)
        resized = cv2.resize(Image, dim, interpolation=cv2.INTER_AREA)
    print(resized.shape)
    return resized


################################################################################################################################################

cam = cv2.VideoCapture(Video_Path)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

model = YOLO(weights)

while True:
    success, img = cam.read()
    image = resize_Video(Image=img, scale_percentage=30)
    results = model(image, True)

    for res in results:
        boxes = res.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Calculate the distance
            distance = round((math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) * 3.28, 2)
            if distance > 2100:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            # centers
            cx1, cy1 = x1 + w // 2, y1 + h // 2

            conf = math.ceil(box.conf[0] * 100) / 100  # get the confidence from box
            Classes = int(box.cls[0])
            currentClass = class_names[Classes]

            Text = f"{currentClass}, {conf}"

            if currentClass == "person" or currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                    or currentClass == "motorbike" and conf > 0.3:
                cvzone.cornerRect(img=image, bbox=(x1, y1, w, h), l=15, t=3, rt=2, colorR=color)

                cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.9, thickness=1,
                                   font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=color)

                cvzone.putTextRect(img=image, text=f"{distance} ft", pos=(max(0, x2 - 70), max(35, y1 - 15)), scale=0.9, thickness=1,
                                   font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=color, offset=1)

    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFf == ord('q'):
        break
