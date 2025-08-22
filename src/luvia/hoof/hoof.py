import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
from skimage.transform import resize
from skimage import io, filters, color
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

class Hoofs:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Hoofs.load_image(image_path)


class Hoof_HThresh(Hoofs):

    @staticmethod
    def apply_binary_threshold(image,max_value=255):
        # Apply binary thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def apply_horizontal_dilation(binary, kernel_size=(50, 40), iterations=1):
        # Apply horizontal dilation to group characters into lines
        #kernel_size = (50,5)
        tall_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated = cv2.dilate(binary, tall_kernel, iterations=iterations)
        return dilated

    @staticmethod
    def find_contours(dilated):
        # Find contours of the dilated image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def draw_bounding_boxes(image, contours):
        # Draw bounding boxes around each detected line
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        line_images = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cropped_line = image[y:y+h, x:x+w]
            line_images.append(cropped_line)
        return output_image, line_images

    @staticmethod
    def extract_lines(image, kernel_size=(50, 10), iterations=1):
        print("KERNEL", kernel_size, "ITERATION", iterations)
        binary = Hoof_HThresh.apply_binary_threshold(image)
        dilated = Hoof_HThresh.apply_horizontal_dilation(binary, kernel_size=kernel_size,
                                                iterations=iterations)
        contours = Hoof_HThresh.find_contours(dilated)
        output_image, line_images = Hoof_HThresh.draw_bounding_boxes(image, contours)
        return output_image, line_images   



class ShorthandSegmenter:

    def __init__(self, min_area_segment=100, dilation_kernel=(25, 5), angle_tolerance=15):
    
        self.min_area = min_area_segment
        self.dilation_kernel = dilation_kernel
        self.angle_tolerance = angle_tolerance
        self.bounding_boxes = []
        self.angles = []
        self.groups_with_angles = []

    def compute_angle(self, component_mask):
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = math.degrees(math.atan2(vy, vx))
            return angle
        return None

    def extract_groups(self, original, filter_boxes=False):
        """
        Extracts shorthand writing groups from an image.
        Returns a list of tuples: (cropped_image, angle)
        """
        if original is None:
            raise ValueError(f"Image at path '{image_path}' could not be loaded.")

        img_height, img_width = original.shape
        _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilation_kernel)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        temp_boxes = []
        temp_groups = []
        temp_angles = []


        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            if filter_boxes == "whole_img":
                # Skip boxes that are nearly the size of the image
                if w >= img_width - 10 and h >= img_height - 10:
                    continue

            if area >= self.min_area:
                cropped = original[y:y+h, x:x+w]
                component_mask = np.zeros_like(binary)
                component_mask[labels == i] = 255
                component_crop = component_mask[y:y+h, x:x+w]
                angle = self.compute_angle(component_crop)
                temp_groups.append(cropped)
                temp_boxes.append((x, y, w, h))
                temp_angles.append(angle)
        if filter_boxes == "inside_box":

            # Filter out boxes that fully contain another box
            def contains(box1, box2):
                x1, y1, w1, h1 = box1
                x2, y2, w2, h2 = box2
                return x1 <= x2 and y1 <= y2 and x1 + w1 >= x2 + w2 and y1 + h1 >= y2 + h2

            filtered_boxes = []
            filtered_groups = []
            filtered_angles = []

            for i, box_i in enumerate(temp_boxes):
                is_enclosing = False
                for j, box_j in enumerate(temp_boxes):
                    if i != j and contains(box_i, box_j):
                        is_enclosing = True
                        break
                if not is_enclosing:
                    filtered_boxes.append(box_i)
                    filtered_groups.append(temp_groups[i])
                    filtered_angles.append(temp_angles[i])

            self.bounding_boxes = filtered_boxes
            self.groups_with_angles = filtered_groups
            self.angles = filtered_angles
        else:
            self.bounding_boxes = temp_boxes
            self.groups_with_angles = temp_groups
            self.angles = temp_angles
        return self.groups_with_angles

    def draw_bounding_boxes(self, image):

        if image is None:
            raise ValueError("Input image is None.")

        if not self.bounding_boxes:
            raise RuntimeError("No bounding boxes found. Run extract_groups() first.")

        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in self.bounding_boxes:
            cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return image_bgr


    def filter_by_angle(self, target_angle):
        """
        Filters extracted groups by angle within the specified tolerance.
        Returns a list of (cropped_image, angle) tuples.
        """
        if not self.groups_with_angles:
            raise RuntimeError("No groups extracted. Run extract_groups() first.")

        filtered = [
            img for img, angle in zip(self.groups_with_angles, self.angles)
            if angle is not None and abs(angle - target_angle) <= self.angle_tolerance
        ]
        return filtered



class Hoof_VThresh:

    @staticmethod
    def binarize_image(image):
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def vertical_projection(binary):
        return np.sum(binary, axis=0)

    @staticmethod
    def gaussian_smoothing(vertical_projection, sigma=6):
        return gaussian_filter1d(vertical_projection, sigma=sigma)

    @staticmethod
    def local_minima_maxima(smoothed_projection):
        minima = argrelextrema(smoothed_projection, np.less)[0]
        maxima = argrelextrema(smoothed_projection, np.greater)[0]
        return minima, maxima

    @staticmethod
    def image_to_color(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def draw_minima_maxima(image_color, minima, maxima):
        # Flatten if minima/maxima are regions (tuples)
        if len(minima) > 0 and isinstance(minima[0], tuple):
            minima = [start for start, _ in minima]
        if len(maxima) > 0 and isinstance(maxima[0], tuple):
            maxima = [start for start, _ in maxima]

        # Draw minima in red
        for x in minima:
            cv2.line(image_color, (int(x), 0), (int(x), image_color.shape[0]), (0, 0, 255), 1)

        # Draw maxima in green
        for x in maxima:
            cv2.line(image_color, (int(x), 0), (int(x), image_color.shape[0]), (0, 255, 0), 1)

        return image_color

    @staticmethod
    def refine_consecutive_minima(minima, separation=5):
        regions = []
        for i in range(len(minima) - 1):
            start = minima[i]
            end = minima[i + 1]
            if end - start > separation:
                regions.append((start, end))
        return regions

    @staticmethod
    def filter_and_extend_minima(minima, maxima, projection, threshold=5):
        # Flatten minima if they are regions (tuples)
        if len(minima) > 0 and isinstance(minima[0], tuple):
            minima = [start for start, _ in minima]

        filtered_minima = []
        for m in minima:
            if all(abs(int(m) - int(M)) > threshold for M in maxima):
                filtered_minima.append(m)

        # Add minima where projection is zero and starts increasing
        for i in range(1, len(projection)):
            if projection[i - 1] == 0 and projection[i] > 0:
                filtered_minima.append(i)

        # Remove duplicates and sort
        return np.array(filtered_minima)


    @staticmethod
    def extend_minima_on_zero_rise(projection):
        minima_candidates = []
        for i in range(1, len(projection)):
            prev_val = projection[i - 1]
            curr_val = projection[i]
            if prev_val == 0 and curr_val > 0:
                if i + 1 < len(projection) and projection[i + 1] >= curr_val:
                    minima_candidates.append(i)
                elif i + 1 == len(projection):
                    minima_candidates.append(i)
        return np.unique(minima_candidates)



    @staticmethod
    def extract_regions(image, regions):
        image_lst = []
        for idx, (start, end) in enumerate(regions):
            char_img = image[:, start:end]
            padded_char = cv2.copyMakeBorder(char_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            image_lst.append(padded_char)
        return image_lst

    @staticmethod
    def vertical_projection_segmentation(image, sigma=6, separation_char=None):
        image_binary = Hoof_VThresh.binarize_image(image)
        vertical_projection_profile = Hoof_VThresh.vertical_projection(image_binary)
        smoothed_projection = Hoof_VThresh.gaussian_smoothing(vertical_projection_profile, sigma)
        minima, maxima = Hoof_VThresh.local_minima_maxima(smoothed_projection)
        if separation_char is not None:
            minima = Hoof_VThresh.filter_and_extend_minima(minima, maxima, smoothed_projection, threshold=separation_char)
            extra_minima = Hoof_VThresh.extend_minima_on_zero_rise(smoothed_projection)
            minima = np.unique(np.concatenate([minima, extra_minima]))
            minima = Hoof_VThresh.refine_consecutive_minima(minima, separation=separation_char)
        image_color = Hoof_VThresh.image_to_color(image)
        image_color = Hoof_VThresh.draw_minima_maxima(image_color, minima, maxima)
        characters = Hoof_VThresh.extract_regions(image, regions=minima)
        return image_color, characters, {
            "projection": smoothed_projection,
            "minima": minima,
            "maxima": maxima}
        
