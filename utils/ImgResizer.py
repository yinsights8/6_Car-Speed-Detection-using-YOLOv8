import cv2

class ImageResizer:
    def __init__(self, scale_percentage=50):
        self.scale_percentage = scale_percentage
        self.width = 640
        self.height = 480

    def resize(self, image):
        height, width, _ = image.shape

        if height == 480 and width == 640:
            resized = image

        else:
            # Resize the frame
            scale_percent = self.scale_percentage
            new_width = int(width * scale_percent / 100)
            new_height = int(height * scale_percent / 100)
            dim = (new_width, new_height)
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized
