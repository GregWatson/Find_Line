import cv2
import numpy as np
import sys
from FL_lib.fl_core import find_start_point, get_adjacent_points
from FL_lib.find_line import find_line

# Finds lines within the given BGR image and returns a list of lines, 
# where each line is represented as a tuple of (points, angle). 
# Points is a list of (x, y) coordinates along the line, 
# and angle is the average angle of the line in radians. 
# The function also returns the grayscale version of the input image 
# with the lines marked in grey. 
# The len_thresh parameter specifies the minimum length of a line to be considered valid. 
# Lines shorter than this threshold will be discarded. 

def find_lines(bgr_img, len_thresh=10, debug=False):
    lines_found = []
    # Convert the image to simple binary format (grayscale)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    #find a starting point for the line
    start_point = find_start_point(gray)

    if not start_point:
        if debug:
            print("ERROR: Unable to find a starting point for the line.")
        sys.exit(1)

    # get list of points adjacent to starting point
    adjacent_points = get_adjacent_points(gray, start_point)
    if len(adjacent_points) == 0:
        if debug:
            print("ERROR: No adjacent points found for the starting point.")
        sys.exit(1)


    if debug:
        print("Starting point:", start_point)
        print("Adjacent points:", adjacent_points)

    points, angle = find_line(start_point, adjacent_points[0], gray, len_thresh, debug)
    if len(points):
        lines_found.append((points, angle))

    return lines_found, gray