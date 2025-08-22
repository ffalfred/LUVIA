import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from skimage.transform import resize
from skimage import io, filters, color
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema   

class Eyes_Contour_Clean:

    @staticmethod
    def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
        #Controls how much the image is smoothed before thresholding.
        #To include more detail, reduce the kernel size (e.g., (3, 3)) or skip blurring.
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def apply_adaptive_threshold(image, block_size=15, C=3):
        #block_size: size of the neighborhood used to calculate threshold.
        #C: constant subtracted from the mean.
        #To include more strokes, try increasing block_size (e.g., 21) or decreasing C (e.g., 1).
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
    
    @staticmethod
    def filter_contours(image, contours, min_area=20, max_area=2000, min_aspect=0.1, max_aspect=10.0, min_vertices=6):
        #Filters out contours that are too small or too large.
        #To include more lines, lower min_area (e.g., 10) and raise max_area (e.g., 3000).
        #Filters based on shape proportions.
        #To include more shapes, widen the range (e.g., min_aspect=0.05, max_aspect=15.0).
        #Filters out simple shapes (like straight lines).
        #To include simpler strokes, reduce this (e.g., min_vertices=3).
        mask = np.zeros_like(image)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if min_area < area < max_area and min_aspect < aspect_ratio < max_aspect and len(approx) >= min_vertices:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    @staticmethod
    def mask_image(image, mask):
        result = np.full_like(image, 255)  # white background
        result[mask == 255] = image[mask == 255]
        return result

    @staticmethod
    def contour_image(image, contours):
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_contours = cv2.drawContours(image_color, contours, -1, (0, 255, 0), 1)
        return image_contours

    @staticmethod
    def extract_original_strokes(image,
                                blur_kernel=(5, 5), blur_sigma=0,
                                block_size=15, C=3,
                                min_area=20, max_area=2000,
                                min_aspect=0.1, max_aspect=10.0,
                                min_vertices=6):
        blurred = Eyes_Contour_Clean.apply_gaussian_blur(image, blur_kernel, blur_sigma)
        thresh = Eyes_Contour_Clean.apply_adaptive_threshold(blurred, block_size, C)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = Eyes_Contour_Clean.filter_contours(image, contours, min_area, max_area, min_aspect, max_aspect, min_vertices)
        return mask




class Eyes_OTSU_Clean:

    @staticmethod
    def apply_median_blur(image, kernel_size=5):
        """
        Apply median blur to reduce noise while preserving edges.
        - kernel_size: Must be odd. Lower values preserve more detail.
        """
        #Purpose: Controls how much noise is removed.+
        #To include more detail: Lower it to 3 or even skip blurring.
        #Effect: Preserves finer strokes that might otherwise be smoothed out.
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def apply_otsu_threshold(image):
        """
        Apply Otsu's thresholding to binarize the image.
        Automatically determines the optimal threshold value.
        """
        #Purpose: Defines sensitivity of edge detection.
        #To include more edges: Lower both values, e.g., 30 and 100.
        #Effect: Captures weaker or thinner strokes.
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def apply_canny_edge_detection(image, threshold1=50, threshold2=150):
        """
        Apply Canny edge detection to find edges.
        - threshold1: Lower bound for edge detection.
        - threshold2: Upper bound for edge detection.
        """
        #Purpose: Filters connected components by size.
        #To include more strokes:
        #Lower cc_min_area to 10 or 5
        #Raise cc_max_area to 3000 or more
        #Effect: Keeps smaller and larger stroke regions.
        return cv2.Canny(image, threshold1, threshold2)

    @staticmethod
    def filter_connected_components(binary_image, min_area=20, max_area=2000):
        """
        Filter connected components based on area.
        - min_area: Minimum pixel area to keep.
        - max_area: Maximum pixel area to keep.
        """
        #Purpose: Filters contours by area.
        #To include more: Same as above—lower min_area, raise max_area.
        #Effect: Accepts more varied stroke sizes.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        mask = np.zeros_like(binary_image)
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                mask[labels == i] = 255
        return mask

    @staticmethod
    def filter_contours_by_shape(image, contours, min_area=20, max_area=2000, min_vertices=5, hu_threshold=0.001):
        """
        Filter contours using area, shape complexity (arcLength), and Hu Moments.
        - min_area: Minimum contour area.
        - max_area: Maximum contour area.
        - min_vertices: Minimum number of vertices in approximated contour.
        - hu_threshold: Minimum value of first Hu Moment to keep contour.
        """
        #Purpose: Filters out simple shapes (like straight lines).
        #To include simpler strokes: Lower to 3 or even 2.
        #Effect: Accepts less complex contours.
        #Purpose: Filters based on shape characteristics.
        #To include more shapes: Lower to 0.0001 or even 0.0.
        #Effect: Accepts more varied and irregular shapes.
        mask = np.zeros_like(image)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area):
                continue
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) < min_vertices:
                continue
            moments = cv2.moments(cnt)
            huMoments = cv2.HuMoments(moments).flatten()
            if huMoments[0] > hu_threshold:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask

    @staticmethod
    def apply_mask_to_image(original_image, mask):
        """
        Apply mask to original image to retain only selected regions.
        Returns image with white background and original strokes preserved.
        """
        result = np.full_like(original_image, 255)
        result[mask == 255] = original_image[mask == 255]
        return result

    @staticmethod
    def extract_shorthand_strokes(image,
                                blur_kernel_size=5,
                                canny_thresh1=50, canny_thresh2=150,
                                cc_min_area=20, cc_max_area=2000,
                                contour_min_area=20, contour_max_area=2000,
                                contour_min_vertices=5, hu_moment_threshold=0.001):
        blurred = Eyes_OTSU_Clean.apply_median_blur(image, blur_kernel_size)
        otsu_thresh = Eyes_OTSU_Clean.apply_otsu_threshold(blurred)
        edges = Eyes_OTSU_Clean.apply_canny_edge_detection(blurred, canny_thresh1, canny_thresh2)
        cc_mask = Eyes_OTSU_Clean.filter_connected_components(otsu_thresh, cc_min_area, cc_max_area)
        contours, _ = cv2.findContours(cc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = Eyes_OTSU_Clean.filter_contours_by_shape(image, contours,
                                            contour_min_area, contour_max_area,
                                            contour_min_vertices, hu_moment_threshold)
        result = Eyes_OTSU_Clean.apply_mask_to_image(image, final_mask)
        return result

