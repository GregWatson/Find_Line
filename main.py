import cv2
import numpy as np
import argparse
import sys
from FL_lib.find_lines import find_lines

def main():
    parser = argparse.ArgumentParser(description="Piece Project CLI")
    # parser.add_argument("-p", "--picture", help="Path to an image file to display", type=str)
    args = parser.parse_args()

    SIZE = 10
    # create a white canvas of size SIZExSIZE
    img = np.zeros((SIZE, SIZE,3), dtype=np.uint8)

    # draw a line
    cv2.line(img, (1,1), (8,8), (255, 255, 255), thickness=1)

    # second line up from 1st line
    cv2.line(img, (8,8), (8,1), (255, 255, 255), thickness=1)

    gray = find_lines(img)

    # SCALE IMAGE TO 500X500
    scaled_img = cv2.resize(gray, (500, 500), interpolation=cv2.INTER_NEAREST)

    # Display the image
    cv2.imshow("Line", scaled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
