import sys

import cv2
import numpy as np
from FL_lib.find_lines import find_lines
from FL_lib.fl_core import get_palette
from FL_lib.find_corners import find_corners

# Find lines in a jigsaw piece outline
def run_test_4(test_params):
    test_num = 4
    print(f"Running test {test_num}...")
    SIZE = 500
    # create a white canvas of size SIZExSIZE
    # img_names = ["Edges/edges_B.png", "Edges/edges_C.png"]
    img_names = ["Edges/edges_B.png", "Edges/edges_C.png", "Edges/edges_D.png"]

    for img_name in img_names:
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
            # print(f"Line {i}: from {points[0]} to {points[-1]}, length = {length:.2f},  angle {np.degrees(angle):.2f} degrees. color {color_names[color_index % len(color_names)]}")
            if length > 15:
                cv2.line(resized_image, start_pt, end_pt, palette[color_index % len(palette)], thickness=3)
                if color_index == len(palette)-1:
                    color_index = 0
                else: color_index += 1
                if dx == 3:
                    dx = -3
                else:
                    dx = 3

        # get corners
        corners = find_corners(lines_found, corner_thresh=50, end_to_end_dist_thresh=20, debug=test_params['debug'])
        if len(corners) < 4:
            print(f"Test {test_num} failed: Expected 4 corners, found {len(corners)}")
            return False
        for i,corner in enumerate(corners[0:4]):
            if test_params['debug']:
                print(f"Corner {corner[0]}: point={corner[1]}, angle_diff={np.degrees(corner[2]):.2f} degrees")
                cv2.circle(resized_image, corner[1], 10, (0, 128, 255), -1)

        all_expected_corners = { "Edges/edges_B.png":[(434, 121), (226, 21), (313, 389), (153, 249)],
                                 "Edges/edges_C.png":[(157,179), (133,326), (322,365), (323,158)],
                                 "Edges/edges_D.png":[(1,1), (1,1), (1,1), (1,1)] 
                              }
        expected_corners = all_expected_corners[img_name]
        
        for i, corner in enumerate(corners[0:4]):
            if not (abs(corner[1][0] - expected_corners[i][0]) <= 2 and abs(corner[1][1] - expected_corners[i][1]) <= 2):
                print(f"*** FAIL ***: Test {test_num} failed: Expected corner at {expected_corners[i]}, found at {corner[1]}")
                return False

        if test_params['debug']:
            cv2.imshow(f"Edge Image {img_name}", resized_image)

    if test_params['debug']:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True
