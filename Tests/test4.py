import sys

import cv2
import numpy as np
from FL_lib.find_lines import find_lines

# Find lines in a jigsaw piece outline
def run_test_4(test_params):
    test_num = 4
    print(f"Running test {test_num}...")
    SIZE = 500
    # create a white canvas of size SIZExSIZE
    img_name = "Edges/edges_B.png"
    WHITE = (255,255,255)

    image = cv2.imread(img_name)
    if image is None:
        print(f"Error: Could not load image from {img_name}. Please check the file path and ensure the image exists.")
        sys.exit(1)
    
    # Resize to SIZExSIZE
    resized_image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

    lines_found, gray = find_lines(image, len_thresh=10, debug=test_params['debug'])

    color = (0, 255, 0)  # Green color for lines
    dx = 3
    for i, line in enumerate(lines_found):
        points, angle = line
        start_pt = (points[0][0] + dx, points[0][1])
        end_pt = (points[-1][0] + dx, points[-1][1])
        length = np.linalg.norm(np.array(points[-1]) - np.array(points[0]))
        print(f"Line {i}: from {points[0]} to {points[-1]}, length = {length:.2f},  angle {np.degrees(angle):.2f} degrees")
        if length > 30:
            cv2.line(resized_image, start_pt, end_pt, color, thickness=3)
            if color[0] != 0:
                color = (0, 255, 0)  # Green
            else: 
                if color[1] != 0:
                    color = (0, 0, 255)  # Blue
                else:
                    color = (255, 0, 0)  # Red
            if dx == 3:
                dx = -3
            else:
                dx = 3
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


