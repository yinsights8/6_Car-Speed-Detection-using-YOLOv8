import cv2


def readVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def saveVideo(OutputvideoFrames, outPutVideoPath):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"{outPutVideoPath}/output.avi", fourcc, 24, (OutputvideoFrames[0].shape[1], OutputvideoFrames[0].shape[0]))
    for frame in OutputvideoFrames:
        out.write(frame)
    out.release()
