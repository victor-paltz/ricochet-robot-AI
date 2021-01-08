from typing import Any, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tools.geometry import angular_dist_modulo, intersection
from tools.kruskal import kruskal_groups


def extract(board: Union[np.ndarray, Any]) -> np.ndarray:

    # extract the lines of the image
    gray = cv2.cvtColor(np.array(board), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, 1*np.pi/180, 200)
    lines = [x[0] for x in lines]
    rhos = [x[0] for x in lines]
    thetas = [x[1] for x in lines]

    # filter roughly the biggest group of orthogonal lines
    keep = set(kruskal_groups(thetas, lambda a, b: angular_dist_modulo(
        a, b, angle=np.pi/2), 1, d_max=30*np.pi/180)[0])
    remove_filter = [t not in keep for t in thetas]
    rhos = [rho for (rho, rem) in zip(rhos, remove_filter) if not rem]
    thetas = [theta for (theta, rem) in zip(thetas, remove_filter) if not rem]

    # label horizontal lines, vertical lines, and outliers
    ori = [theta % np.pi for theta in thetas]
    ori_reshape = np.array(ori).reshape(-1, 1)
    s1, s2 = kruskal_groups(ori, lambda a, b: angular_dist_modulo(
        a, b, angle=np.pi), 2, d_max=3*np.pi/180)[:2]
    remove_filter_group = [((x not in s1) and (x not in s2)) for x in ori]
    predicted_classes = [1 if x in s2 else 0 for x in ori]
    cluster_centers = list(s1)[0], list(s2)[0]

    # remove useless lines
    remove_filter = remove_filter_group
    new_class = [c for (c, rem) in zip(
        predicted_classes, remove_filter) if not rem]
    new_rhos = [rho for (rho, rem) in zip(rhos, remove_filter) if not rem]
    new_thetas = [theta for (theta, rem) in zip(
        thetas, remove_filter) if not rem]

    # Extract the boundary lines of the board game
    final_lines = [(np.float("inf"), -1), (-np.float("inf"), -1),
                   (np.float("inf"), -1), (-np.float("inf"), -1)]
    vals = [np.float("inf"), -np.float("inf"),
            np.float("inf"), -np.float("inf")]
    for r, t, c in zip(new_rhos, new_thetas, new_class):
        if c == 0:
            d = r
            if abs(t - cluster_centers[c]) > np.pi/2:
                d *= -1
            if d < vals[0]:
                final_lines[0] = (r, t)
                vals[0] = d
            if d > vals[1]:
                final_lines[1] = (r, t)
                vals[1] = d
        else:
            d = r
            if abs(t - cluster_centers[c]) > np.pi/2:
                d *= -1
            if d < vals[2]:
                final_lines[2] = (r, t)
                vals[2] = d
            if d > vals[3]:
                final_lines[3] = (r, t)
                vals[3] = d

    # extract corners of the board game
    pts = []
    pts.append(intersection(final_lines[2], final_lines[0]))
    pts.append(intersection(final_lines[1], final_lines[2]))
    pts.append(intersection(final_lines[0], final_lines[3]))
    pts.append(intersection(final_lines[3], final_lines[1]))

    # extract board
    p1 = [min((pt[0]+pt[1], pt) for pt in pts)[1]]
    p4 = [max((pt[0]+pt[1], pt) for pt in pts)[1]]
    p_res = list(
        sorted([pt for pt in pts if pt != p1[0] and pt != p4[0]], reverse=True))
    pts_ordered = p1 + p_res + p4
    pts1 = np.float32(pts_ordered)
    size = 600
    pts2 = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(np.array(plateau), M, (size, size))

    return dst


if __name__ == "__main__":
    plateau = Image.open("images/plateau4.jpeg")

    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.imshow(plateau), plt.title('Input')
    plt.subplot(122), plt.imshow(extract(plateau)), plt.title('Output')
    plt.show()
