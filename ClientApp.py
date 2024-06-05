import cv2
from utils.ImgResizer import ImageResizer
from flask import Flask, redirect, url_for, render_template, request, Response, flash
from werkzeug.utils import secure_filename  # it will replace or make changes in the original file names to secure it
from werkzeug.exceptions import RequestEntityTooLarge
from ultralytics import YOLO
from utils.detections import Detector
from werkzeug.utils import secure_filename, send_from_directory
import time
import os


# create a object of flask
app = Flask(__name__)

app.config["UPLOAD_DIRECTORY"] = "uploads/"  # global file directory name to store files
app.config["MAX_CONTENT_LENGTH"] = 950 * 1024 *1024 # 30MB
app.config["ALLOWED_EXTENSIONS"] = ['.jpg', '.jpeg', '.png', '.mp4']
app.config["SECRET_KEY"] = "secretkey"

model_path = "Weights/yolov8m.pt"

uploadFile = []
    

# create app
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ComputerVision")
def ComputerVision():
    return render_template("ComputerVision.html")

# for object detection
@app.route("/objectDetection")
def objectDetection():
    return render_template("ObjectDetection.html")


# upload files and predictions
@app.route("/objectDetection", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        file = request.files["uploadedFile"]
        extensions = os.path.splitext(file.filename)[1]
        try:
            # to avoid uploading nothing
            if file:
                
                if extensions not in app.config["ALLOWED_EXTENSIONS"]:
                    flash("Wrong file selected\nallowed only IMAGES and VIDEOS")
                    return redirect(url_for('objectDetection'))
                    # return "Wrong file selected\nallowed only IMAGES and VIDEOS"

                # creating a path to the uploaded file
                file_name = secure_filename(file.filename)
                directory = app.config["UPLOAD_DIRECTORY"]    # global directory name
                basename = os.path.dirname(__file__)          # basename
                Path = os.path.join(directory, file_name)     # full path
                fullPath = os.path.join(basename,Path)        # E:\My_Model\3._Object_Detection\6_Car Speed Detector\uploads/highway.mp4
                uploadFile.append(fullPath) 
                file.save(fullPath)
                flash("FIle uploaded Successfully!")
                # return redirect(url_for('objectDetection'))
                fileEXT = file.filename.split(".")[1]
                
                if fileEXT in app.config["ALLOWED_EXTENSIONS"][:3]:
                    img = cv2.imread(fullPath)

                    # Perform the detection
                    model = YOLO(model_path)
                    detections =  model(img, save=True) 
                    return display(f.filename)

                # elif fileEXT == 'mp4':
                #     video_path = fullPath  # replace with your video path
                #     detections = Detector(video_path, model_path)
                #     detections.ouputVideo()
                #     return video_feed()

                # for when running on live video
                elif fileEXT == 'mp4':
                    return video_feed()


                folder_path = 'runs/detect'
                subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
                latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
                image_path = folder_path+'/'+latest_subfolder+'/'+ file.filename 
                return render_template('index.html', image_path=image_path)
                
        except RequestEntityTooLarge:
            flash("file too large")




#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    directory = folder_path+'/'+latest_subfolder    
    print("printing directory: ",directory) 
    files = os.listdir(directory)
    latest_file = files[0]
    
    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

    else:
        return "Invalid file format"



# get frames on output video
def get_output_frame():

    currentFolder = os.getcwd()
    print("Current folder::::::::::::::::: " + currentFolder)
    video_path = "output_video.mp4"

    cam  = cv2.VideoCapture(video_path)
    while True:
        succ, img = cam.read()

        if not succ: 
            break
        
        else:
            ret, buffer = cv2.imencode(".jpg", img)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

# get frames on 
def get_frame():
    video_path = "".join(uploadFile[-1])
    # class_names = readClass(classFile)
    detect = Detector(videoPath=video_path, modelPath=model_path, scalPercent=60)
    while True:
        frame =  detect.outputFrames()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)



# for live stream detection
@app.route('/video_feed')
def video_feed():
    print("running function")
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)