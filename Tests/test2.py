import cv2
import numpy as np
from FL_lib.find_lines import find_lines

# simple straight lines with several single point interference.
def run_test_2(test_params):
    test_num = 2
    print(f"Running test {test_num}...")
    SIZE = test_params['CANVAS_SIZE']
    # create a white canvas of size SIZExSIZE
    img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)

    l_start = (1,1)
    for y in range(1,SIZE-2):
        # reset image to blank
        img.fill(0)

        l_end = (SIZE-2,y)
        if test_params['LOG'] or test_params['debug']:
            print(f"Testing line from {l_start} to {l_end}...")

        # draw a line from L_START to L_END
        cv2.line(img, l_start, l_end, (255, 255, 255), thickness=1)

        # draw line interference
        cv2.line(img, (1,2),(2,1), (255, 255, 255), thickness=1)

        lines_found, _ = find_lines(img, test_params['LEN_THRESH'], debug=test_params['debug'])

        if len(lines_found) != 1:
            print(f"Test {test_num} failed: Expected 1 line, found {len(lines_found)}")
            return False
        points, _ = lines_found[0]
        if points[0] != l_start or points[-1] != l_end:
            print(f"Test {test_num} failed: Expected line from {l_start} to {l_end}, found from {points[0]} to {points[-1]}")
            return False
    return True


