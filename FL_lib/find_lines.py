import cv2
import numpy as np
import sys
from FL_lib.fl_core import find_start_point, get_adjacent_points
from FL_lib.find_line import find_line

def find_lines(bgr_img):
    # Convert the image to simple binary format (grayscale)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    #find a starting point for the line
    start_point = find_start_point(gray)

    if not start_point:
        print("ERROR: Unable to find a starting point for the line.")
        sys.exit(1)

    # get list of points adjacent to starting point
    adjacent_points = get_adjacent_points(gray, start_point)
    if len(adjacent_points) == 0:
        print("ERROR: No adjacent points found for the starting point.")
        sys.exit(1)


    print("Starting point:", start_point)
    print("Adjacent points:", adjacent_points)

    find_line(start_point, adjacent_points[0], gray)

    return gray