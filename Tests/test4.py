import sys

import cv2
import numpy as np
from FL_lib import find_rotation, rotate_image
from FL_lib.fl_core import draw_lines_on_color_image, find_piece_center, get_distance_between_2_points, get_palette, rotate_point
from FL_lib.find_corners import find_corners
from FL_lib.pre_proc_image import pre_process_image
from FL_lib.find_rotation import find_rotation
from FL_lib.rotate_image import rotate_image

# Find lines in a jigsaw piece outline
def run_test_4(test_params):
    test_num = 4
    print(f"Running test {test_num}...")
    SIZE = 500
    # create a white canvas of size SIZExSIZE
    # img_names = ["Edges/edges_B.png", "Edges/edges_C.png"]
    img_names = ["Edges/edges_B.png", "Edges/edges_C.png", "Edges/edges_D.png"]

    palette, color_names = get_palette(palette_size=6)

    for img_name in img_names:
        image = cv2.imread(img_name)
        if image is None:
            print(f"Error: Could not load image from {img_name}. Please check the file path and ensure the image exists.")
            sys.exit(1)
        min_length = test_params['LEN_THRESH']

        cx, cy = find_piece_center(image)
        rot_angle, cx, cy, lines = find_rotation(image, cx, cy)

        draw_lines_on_color_image(image, lines, palette)
        cv2.imshow(f"Orig Image {img_name}", image)

        print(f"Found {len(lines)} lines. Rotation angle: {np.degrees(rot_angle):.2f} degrees")

        rotated_image = rotate_image(image, rot_angle, cx, cy)
        rotated_lines = []
        for line in lines:
            p1_rot = rotate_point(line[0], (cx,cy), rot_angle)
            p2_rot = rotate_point(line[1], (cx,cy), rot_angle)
            rotated_lines.append((p1_rot, p2_rot))


        draw_lines_on_color_image(rotated_image, rotated_lines, palette, dx=3)

        # get corners
        corners = find_corners(rotated_lines, corner_thresh=50, end_to_end_dist_thresh=20, debug=test_params['debug'])

        if len(corners) < 4:
            print(f"Test {test_num} failed: Expected 4 corners, found {len(corners)}")
            return False
        for i,corner in enumerate(corners[0:4]):
            if test_params['debug']:
                print(f"Corner-ness {corner[0]}, point={corner[1]}, angle_diff={np.degrees(corner[2]):.2f} degrees")
                cv2.circle(rotated_image, corner[1], 10, (0, 128, 255), -1)

        if test_params['debug']:
            cv2.imshow(f"Edge Image {img_name}", rotated_image)
            cv2.waitKey(0)


        all_expected_corners = { "Edges/edges_B.png":[(227,20), (313, 389), (434, 121), (153, 249)],
                                 "Edges/edges_C.png":[(157,179), (133,326), (322,365), (323,158)],
                                 "Edges/edges_D.png":[(1,1), (1,1), (1,1), (1,1)] 
                              }
        expected_corners = all_expected_corners[img_name]
        
        for i, corner in enumerate(corners[0:4]):
            corner_x = int(corner[1][0])
            corner_y = int(corner[1][1])
            if not (abs(corner_x - expected_corners[i][0]) <= 2 and abs(corner_y - expected_corners[i][1]) <= 2):
                print(f"*** FAIL ***: Test {test_num} failed: Image:{img_name} Expected corner index {i} at {expected_corners[i]}, found at {corner_x}, {corner_y}")
                return False

    if test_params['debug']:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True
