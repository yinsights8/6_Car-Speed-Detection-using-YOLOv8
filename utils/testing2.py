import cv2
import cvzone
import math
from utils.sort import *
from utils.ImgResizer import ImageResizer
from ultralytics import YOLO

class Detector:
    def __init__(self, videoPath, modelPath, scalPercent = 50):
        self.videoPath = videoPath
        self.modelPath = modelPath
        self.scalPercent = scalPercent

        self.model = YOLO(self.modelPath)
        self.cam = cv2.VideoCapture(self.videoPath)

        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        self.myColor = (255, 0, 255)

        self.L1_limits = [60, 248, 924, 248]  #191
        self.L2_limits = [0, 346, 1135, 346]

        self.red_line_y = 248
        self.blue_line_y = 346
        self.offset = 7

        self.Dwn = dict()
        self.Up = dict()

        self.CountUp = []
        self.CountDwn = []
        self.classNames = "utils/coco.names"
        self.class_names = self.readClass(self.classNames)


        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.new_width = int(self.frame_width * self.scalPercent / 100)
        self.new_height = int(self.frame_height * self.scalPercent / 100)

        self.out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.new_width, self.new_height))

    # video will automatically turn off after closing website
    def __del__(self):
        return self.cam.release()
    
    # Method for reading classes
    def readClass(self, classFile):
        """
        This function will read the class names
        """
        with open(classFile, 'r') as f:
            self.classList = f.read().split()
        return self.classList

    def outputFrames(self):
        """
        This function is use to display the predictions on live video
        This function can be use when predicting on live fotage
        """
        # while True:
        success, img = self.cam.read()
        image = ImageResizer().resize(image=img)
        results = self.model(image, True)

        detections = np.empty((0, 5))

        for res in results:
            boxes = res.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil(box.conf[0] * 100) / 100  # get the confidence from box
                Classes = int(box.cls[0])
                currentClass = self.class_names[Classes]

                if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                        or currentClass == "motorbike" and conf > 0.3:
                    CurrentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((CurrentArray, detections))

            # update the tracker
        resutlTracker = self.tracker.update(dets=detections)

        cv2.line(img=image, pt1=(self.L1_limits[0], self.L1_limits[1]), pt2=(self.L1_limits[2], self.L1_limits[3]), color=(0, 0, 255), thickness=1)  # L1
        cv2.line(img=image, pt1=(self.L2_limits[0], self.L2_limits[1]), pt2=(self.L2_limits[2], self.L2_limits[3]), color=(255, 0, 0), thickness=1)  # L2

        for result in resutlTracker:
            x1, y1, x2, y2, id = result
            w, h = x2 - x1, y2 - y1
            x1, y1, w, h, id = int(x1), int(y1), int(w), int(h), int(id)
            cx, cy = x1 + w // 2, y1 + h // 2

            # When cars going from red line to blue line
            if self.red_line_y < (cy + self.offset) and self.red_line_y > (cy - self.offset):
                self.Dwn[id] = time.time()
            if id in self.Dwn:
                if self.blue_line_y < (cy + self.offset) and self.blue_line_y > (cy - self.offset):
                    elapsedTime1 = time.time() - self.Dwn[id]
                    if self.CountDwn.count(id) == 0:
                        self.CountDwn.append(id)
                        distance = 10
                        speedMs = distance / elapsedTime1
                        SpeedKmph = round(speedMs * 3.9, 2)

                        Text = f"ID: {id} Speed: {SpeedKmph} Kmh"
                        cv2.circle(image, (cx, cy), 4, (50, 50, 255), cv2.FILLED)
                        cv2.line(img=image, pt1=(self.L2_limits[0], self.L2_limits[1]), pt2=(self.L2_limits[2], self.L2_limits[3]), color=(0, 255, 0),
                                thickness=2)
                        cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.5, thickness=1,
                                        font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=self.myColor)

            # When cars going from Blue line to Red line
            if self.blue_line_y < (cy + self.offset) and self.blue_line_y > (cy - self.offset):
                self.Up[id] = time.time()
            if id in self.Up:
                if self.red_line_y < (cy + self.offset) and self.red_line_y > (cy - self.offset):
                    elapsedTime2 = time.time() - self.Up[id]
                    if self.CountUp.count(id) == 0:
                        self.CountUp.append(id)
                        distance1 = 10
                        speedMs = distance1 / elapsedTime2
                        SpeedKmph = round(speedMs * 3.9, 2)

                        Text = f"ID: {id} Speed: {SpeedKmph} Kmh"
                        cv2.circle(image, (cx, cy), 4, (50, 50, 255), cv2.FILLED)
                        cv2.line(img=image, pt1=(self.L2_limits[0], self.L2_limits[1]), pt2=(self.L2_limits[2], self.L2_limits[3]), color=(0, 255, 0),
                                thickness=2)
                        cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.5, thickness=1,
                                        font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=self.myColor)

        cvzone.putTextRect(image, text=f"CountDown = {len(self.CountDwn)}", pos=(40, 30), scale=0.5,
                        thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(255, 0, 255))
        cvzone.putTextRect(image, text=f"CountUp = {len(self.CountUp)}", pos=(40, 50), scale=0.5,
                        thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(255, 0, 255))


    ####################################################################################################################
        self.out.write(image)
        ret, buffer = cv2.imencode(".jpg", image)
        frame = buffer.tobytes()

        return frame

    def ouputVideo(self):
        """
        This function is use to display the output video
        This function can be use when predicting on recoreded fotage
        """
        while True:
            success, img = self.cam.read()
            image = ImageResizer().resize(image=img)
            results = self.model(image, True)

            detections = np.empty((0, 5))

            for res in results:
                boxes = res.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil(box.conf[0] * 100) / 100  # get the confidence from box
                    Classes = int(box.cls[0])
                    currentClass = self.class_names[Classes]

                    if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
                            or currentClass == "motorbike" and conf > 0.3:
                        CurrentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((CurrentArray, detections))

                # update the tracker
            resutlTracker = self.tracker.update(dets=detections)

            cv2.line(img=image, pt1=(self.L1_limits[0], self.L1_limits[1]), pt2=(self.L1_limits[2], self.L1_limits[3]), color=(0, 0, 255), thickness=1)  # L1
            cv2.line(img=image, pt1=(self.L2_limits[0], self.L2_limits[1]), pt2=(self.L2_limits[2], self.L2_limits[3]), color=(255, 0, 0), thickness=1)  # L2

            for result in resutlTracker:
                x1, y1, x2, y2, id = result
                w, h = x2 - x1, y2 - y1
                x1, y1, w, h, id = int(x1), int(y1), int(w), int(h), int(id)
                cx, cy = x1 + w // 2, y1 + h // 2

                # When cars going from red line to blue line
                if self.red_line_y < (cy + self.offset) and self.red_line_y > (cy - self.offset):
                    self.Dwn[id] = time.time()
                if id in self.Dwn:
                    if self.blue_line_y < (cy + self.offset) and self.blue_line_y > (cy - self.offset):
                        elapsedTime1 = time.time() - self.Dwn[id]
                        if self.CountDwn.count(id) == 0:
                            self.CountDwn.append(id)
                            distance = 10
                            speedMs = distance / elapsedTime1
                            SpeedKmph = round(speedMs * 3.9, 2)

                            Text = f"ID: {id} Speed: {SpeedKmph} Kmh"
                            cv2.circle(image, (cx, cy), 4, (50, 50, 255), cv2.FILLED)
                            cv2.line(img=image, pt1=(self.L2_limits[0], self.L2_limits[1]), pt2=(self.L2_limits[2], self.L2_limits[3]), color=(0, 255, 0),
                                    thickness=2)
                            cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.5, thickness=1,
                                            font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=self.myColor)

                # When cars going from Blue line to Red line
                if self.blue_line_y < (cy + self.offset) and self.blue_line_y > (cy - self.offset):
                    self.Up[id] = time.time()
                if id in self.Up:
                    if self.red_line_y < (cy + self.offset) and self.red_line_y > (cy - self.offset):
                        elapsedTime2 = time.time() - self.Up[id]
                        if self.CountUp.count(id) == 0:
                            self.CountUp.append(id)
                            distance1 = 10
                            speedMs = distance1 / elapsedTime2
                            SpeedKmph = round(speedMs * 3.9, 2)

                            Text = f"ID: {id} Speed: {SpeedKmph} Kmh"
                            cv2.circle(image, (cx, cy), 4, (50, 50, 255), cv2.FILLED)
                            cv2.line(img=image, pt1=(self.L2_limits[0], self.L2_limits[1]), pt2=(self.L2_limits[2], self.L2_limits[3]), color=(0, 255, 0),
                                    thickness=2)
                            cvzone.putTextRect(img=image, text=Text, pos=(max(35, x1), max(35, y1 - 15)), scale=0.5, thickness=1,
                                            font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=self.myColor)

            cvzone.putTextRect(image, text=f"CountDown = {len(self.CountDwn)}", pos=(40, 30), scale=0.5,
                            thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(255, 0, 255))
            cvzone.putTextRect(image, text=f"CountUp = {len(self.CountUp)}", pos=(40, 50), scale=0.5,
                            thickness=1, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, colorR=(255, 0, 255))

            # Write the processed frame to the video
            self.out.write(image)  # Pass the original image to out.write()

            cv2.imshow("Video", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release VideoCapture and VideoWriter
        self.cam.release()
        self.out.release()
        cv2.destroyAllWindows()