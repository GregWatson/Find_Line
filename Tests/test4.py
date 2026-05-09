import sys

import cv2
import numpy as np
from FL_lib.find_lines import find_lines
from FL_lib.fl_core import get_palette

# Find lines in a jigsaw piece outline
def run_test_4(test_params):
    test_num = 4
    print(f"Running test {test_num}...")
    SIZE = 500
    # create a white canvas of size SIZExSIZE
    img_name = "Edges/edges_B.png"

    image = cv2.imread(img_name)
    if image is None:
        print(f"Error: Could not load image from {img_name}. Please check the file path and ensure the image exists.")
        sys.exit(1)
    min_length = test_params['LEN_THRESH']
    min_length = 15

    lines_found, _ = find_lines(image, len_thresh=min_length, debug=test_params['debug'])
    if (test_params['debug']):
        print(f"-- Found {len(lines_found)} lines in the image with length of at least {min_length} pixels.")
    palette, color_names = get_palette(palette_size=6)
    color_index = 0
    dx = 3
    # Resize to SIZExSIZE
    resized_image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    for i, line in enumerate(lines_found):
        points, angle = line
        start_pt = (points[0][0] + dx, points[0][1])
        end_pt = (points[-1][0] + dx, points[-1][1])
        length = np.linalg.norm(np.array(points[-1]) - np.array(points[0]))
        print(f"Line {i}: from {points[0]} to {points[-1]}, length = {length:.2f},  angle {np.degrees(angle):.2f} degrees. color {color_names[color_index % len(color_names)]}")
        if length > 15:
            cv2.line(resized_image, start_pt, end_pt, palette[color_index % len(palette)], thickness=3)
            if color_index == len(palette)-1:
                color_index = 0
            else: color_index += 1
            if dx == 3:
                dx = -3
            else:
                dx = 3
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return True


