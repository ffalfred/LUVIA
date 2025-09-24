import cv2
import numpy as np

class ImageUtils:

    @staticmethod
    def load_image(image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    @staticmethod
    def reduce_quality(image, new_width=350):
        # Calculate the new height to maintain aspect ratio
        height, width = image.shape[:2]
        aspect_ratio = height / width
        new_height = int(new_width * aspect_ratio)

        # Resize the image
        resized_img = cv2.resize(image, (new_width, new_height))
        print(resized_img)
        return resized_img
    
    @staticmethod
    def add_canvas(image, canvas_size=300):
        canvas_height = image.shape[0] + canvas_size
        canvas_width = image.shape[1] + canvas_size
        # Create a white canvas
        white_canvas = np.ones((canvas_height, canvas_width), dtype=np.uint8) * 255
        # Compute top-left corner to center the image
        y_offset = (canvas_height - image.shape[0]) // 2
        x_offset = (canvas_width - image.shape[1]) // 2
        # Paste the image onto the canvas
        white_canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        return white_canvas

    @staticmethod
    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute the new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust the rotation matrix to take into account translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation with the new dimensions
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated

    @staticmethod
    def invert_image(image):
        inverted = cv2.bitwise_not(image)
        return inverted


    @staticmethod
    def rotate_image_crop(image, angle):
        # Get image dimensions
        (h, w) = image.shape[:2]
        # Get the center of the image
        center = (w // 2, h // 2)

        # Compute the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    @staticmethod
    def deskew_image(image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed
