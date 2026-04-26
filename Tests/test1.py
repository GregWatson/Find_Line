#tests
import cv2
import numpy as np
from FL_lib.find_line import find_line


# simple straight lines with no interference.
def run_test_1(test_params):
    print("Running test 1...")
    SIZE = test_params['CANVAS_SIZE']
    # create a white canvas of size SIZExSIZE
    img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)

    for l_start in [(1,1), (1,SIZE-2), (SIZE-2,1), (SIZE-2,SIZE-2)]:
        for y in range(1,SIZE-2):
            # reset image to blank
            img.fill(0)

            l_end = (SIZE-1-l_start[0],y)
            if test_params['LOG'] or test_params['debug']:
                print(f"Testing line from {l_start} to {l_end}...")

            # draw a line from L_START to L_END
            cv2.line(img, l_start, l_end, (255, 255, 255), thickness=1)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            points, _ = find_line(l_start, gray, test_params['LEN_THRESH'], debug=test_params['debug'])

            if len(points) <  test_params['LEN_THRESH']:
                print(f"*** FAIL ***: Test 1 failed: Expected at least {test_params['LEN_THRESH']} points, found {len(points)}")
                return False
            if (points[0] != l_start or points[-1] != l_end) and (points[0] != l_end or points[-1] != l_start):
                print(f"*** FAIL ***: Test 1 failed: Expected line from {l_start} to {l_end}, found from {points[0]} to {points[-1]}")
                return False


    return True


