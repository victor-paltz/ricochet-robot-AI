from enum import IntEnum, unique
from typing import Any, Dict, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tools.geometry import angular_dist_modulo, intersection
from tools.kruskal import kruskal_groups


@unique
class Orientation(IntEnum):
    TOP_LEFT = 0
    BOTTOM_LEFT = 1
    BOTTOM_RIGHT = 2
    TOP_RIGHT = 3


def extract_board(board: Union[np.ndarray, Any], square_size: int = 512) -> np.ndarray:
    """
    Extract the board game image from a picture
    """

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

    # extract board first version
    dst = perspective_transform(pts, board, square_size)

    # refine board extraction by finding the board corners
    final_board = refine_board_extraction(dst, n_times=1)

    return final_board


def perspective_transform(points, img, square_size):

    p1 = [min((pt[0]+pt[1], pt) for pt in points)[1]]
    p4 = [max((pt[0]+pt[1], pt) for pt in points)[1]]
    p_res = list(
        sorted([pt for pt in points if pt != p1[0] and pt != p4[0]], reverse=True))
    pts_ordered = p1 + p_res + p4
    # pts_ordered[1] = pts_ordered[1][0]+1, pts_ordered[1][1]
    # pts_ordered[2] = pts_ordered[2][0], pts_ordered[2][1]+1
    # pts_ordered[3] = pts_ordered[3][0]+1, pts_ordered[3][1]+1
    pts1 = np.float32(pts_ordered)
    pts2 = np.float32([[0, 0], [square_size, 0], [0, square_size],
                       [square_size, square_size]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(np.array(img), M, (square_size, square_size))

    return dst


def refine_board_extraction(board: np.ndarray, n_times: int = 1) -> np.ndarray:

    output = board.copy()

    for _ in range(n_times):
        new_corners = list(extract_all_corners(board).values())
        output = perspective_transform(new_corners, output, 512)

    return output


def get_corner_coord(corner_image: np.ndarray, angle_type: Orientation, padding_space: int = 6) -> Tuple[int, int]:

    gray = cv2.cvtColor(corner_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 13, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 15, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges2 = cv2.dilate(thresh, kernel, iterations=1)
    edges2 = cv2.erode(edges2, kernel, iterations=1)
    edges2 = np.pad(edges2, padding_space,
                    mode='constant', constant_values=255)

    rot_dict = {Orientation.TOP_LEFT: 0,
                Orientation.BOTTOM_LEFT: 1,
                Orientation.BOTTOM_RIGHT: 2,
                Orientation.TOP_RIGHT: 3}

    template = np.zeros(
        (corner_image.shape[0]//2+padding_space,
         corner_image.shape[1]//2+padding_space), np.uint8)

    for i in range(padding_space):
        template[i, :] = 1
        template[:, i] = 1

    template = np.rot90(template, rot_dict[angle_type])

    res = cv2.matchTemplate(edges2, template, cv2.TM_CCOEFF)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    coord = max_loc[::-1]  # h, w order for coord

    if angle_type is Orientation.BOTTOM_LEFT:
        coord = coord[0]+template.shape[0]-2*padding_space, coord[1]
    elif angle_type is Orientation.BOTTOM_RIGHT:
        coord = coord[0]+template.shape[0]-2 * \
            padding_space, coord[1]+template.shape[1]-2*padding_space
    elif angle_type is Orientation.TOP_RIGHT:
        coord = coord[0], coord[1]+template.shape[1]-2*padding_space

    return coord[::-1]  # w, h


def extract_all_corners(board: np.ndarray) -> Dict[Orientation, Tuple[int, int]]:

    h, w = board.shape[:2]
    corner_size = 48
    coords = {}
    coords[Orientation.TOP_LEFT] = get_corner_coord(
        board[:corner_size, :corner_size, :], Orientation.TOP_LEFT)
    tmp = get_corner_coord(
        board[-corner_size:, :corner_size, :], Orientation.BOTTOM_LEFT)
    coords[Orientation.BOTTOM_LEFT] = tmp[0], h - corner_size + tmp[1]
    tmp = get_corner_coord(
        board[-corner_size:, -corner_size:, :], Orientation.BOTTOM_RIGHT)
    coords[Orientation.BOTTOM_RIGHT] = w - \
        corner_size + tmp[0], h - corner_size + tmp[1]
    tmp = get_corner_coord(
        board[:corner_size, -corner_size:, :], Orientation.TOP_RIGHT)
    coords[Orientation.TOP_RIGHT] = w - corner_size + tmp[0], tmp[1]

    return coords


if __name__ == "__main__":

    import time
    board = Image.open("images/plateau2.jpeg")
    start_time = time.time()
    extracted_board = extract_board(board)
    print(f"extracted board in {(time.time()-start_time)*1000:.4} ms")

    plt.figure(figsize=(6, 3))
    plt.subplot(121), plt.imshow(board), plt.title('Input')
    plt.subplot(122), plt.imshow(extracted_board), plt.title('Output')
    plt.show()
