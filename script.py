import math

import cv2
import numpy as np

def draw_triangle(img, x, y, orientation, edgeLength):
    # Given orientation of triangle, center point (a,b), and edge length, this function draws lines for each pair of possible
    # corner position, forming the triangle.
    # orientation is in degrees.
    radians = math.radians(orientation)
    angles = [radians + i * (2 * math.pi / 3) for i in range(3)]
    points = []

    # Distance between center point and current corner position
    d = math.floor(edgeLength * math.sqrt(3) / 3)

    for a in angles:
        currx = round(d * math.cos(a)) + x
        curry = round(d * math.sin(a)) + y
        points.append((currx, curry))

    #
    for p in points:
        for p2 in points:
            img = cv2.line(img, p, p2, (0, 255, 0), thickness=1)
    return img

def run_script(img, edgeLength, canny_l, canny_h, center_step, angle_step, n):
    # Votes for triangle center point by travling on lines that are parallel to the edge's direction (perpendicular to
    # the gradient direction) edgeLength/2 to the left, and edgeLength/2 to the right

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = round(edgeLength * math.sqrt(3) / 6)

    rows, cols = img.shape[:2]
    # print(rows, cols)

    canny = cv2.Canny(img_grey, canny_l, canny_h)

    # Calculate x, y gradients of img
    edges_x = cv2.Sobel(canny, cv2.CV_32F, 1, 0, ksize=3)  # ksize = -1 -> Scharr Filter
    edges_y = cv2.Sobel(canny, cv2.CV_32F, 0, 1, ksize=3)

    votes = {}
    for row in range(rows):
        for col in range(cols):
            if canny[row][col] == 255:
                if edges_x[row, col] == 0:
                    theta = 0
                elif edges_y[row, col] == 0:
                    theta = np.pi / 2
                else:
                    theta = np.arctan2(edges_y[row, col], edges_x[row, col])

                # Given edge point and it's direction, our possible center is either that direction or the opposite,
                # thus the math.pi addition (180 degrees)
                thetas = [theta, theta + math.pi]

                # For each direction above, we go d = edgeLength*sqrt(3)/6 steps to that direction
                for t in thetas:
                    xnew = round(d * math.sin(t)) + row
                    ynew = round(d * math.cos(t)) + col
                    if not 0 <= xnew < rows or not 0 <= ynew < cols:
                        continue
                    t2 = [(math.pi / 2) + t, t - (math.pi / 2)]
                    for t3 in t2:
                        for d2 in range(math.ceil(edgeLength / 2)):
                            currx = round(d2 * math.sin(t3)) + xnew
                            curry = round(d2 * math.cos(t3)) + ynew
                            currx = math.floor(currx / center_step) * center_step + math.floor(center_step / 2)
                            curry = math.floor(curry / center_step) * center_step + math.floor(center_step / 2)
                            if not 0 <= currx < rows or not 0 <= curry < cols:
                                continue
                            possible_triangle = (currx, curry, math.floor(
                                math.degrees(t % (2 * math.pi / 3)) / angle_step) * angle_step + angle_step / 2)
                            if possible_triangle not in votes:
                                votes[possible_triangle] = 1
                            else:
                                votes[possible_triangle] += 1
                                # print(votes)

    # Sort all the votes in decreasing order and take top n
    sorteddv = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1], reverse=True)}
    counter = 0
    for vote in sorteddv:
        if counter >= n:
            break

        counter += 1
        img = draw_triangle(img, vote[1], vote[0], vote[2], edgeLength)
    cv2.imshow('Canny', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    #change the parameters to get better results per image.
    img= cv2.imread("./input/triangles_1/image002.jpg") #anquier an image from path
    edgeLength=11 #length of edges on equilateral triangle, same size for every edge.
    canny_l=350 #lower canny edge detection threshold
    canny_h=700 #higher canny edge detection threshold
    center_step=3 #shap center quantization value
    angle_step=20 #shap angle quantization value
    n=200 #number of top votes to take
    run_script(img, edgeLength, canny_l, canny_h, center_step, angle_step, n)