from parameters import Parameters
from PIL import Image, ImageDraw
import numpy as np
import math


def process_image(info_about_photo, img_size, canvas_r):

    img_height = img_size["bottom"] - img_size["top"] + 1
    img_width = img_size["right"] - img_size["left"] + 1

    correct_size = max(img_height, img_width)

    r = int((correct_size * Parameters.RADIUS_PERCENTAGE) / (100 - 2 * Parameters.RADIUS_PERCENTAGE))
    diff_r = r - canvas_r

    image = Image.new("L", (Parameters.WIDTH + 2 * diff_r, Parameters.HEIGHT + 2 * diff_r), 255)
    draw = ImageDraw.Draw(image)

    number_of_pixels = len(info_about_photo)

    for i in range(len(info_about_photo)):

        info = info_about_photo[len(info_about_photo) - 1 - i]

        x = info[0] + diff_r
        y = info[1] + diff_r
        index = info[2]

        color = int((index / number_of_pixels) * (Parameters.MAX_COLOR - Parameters.MIN_COLOR) + Parameters.MIN_COLOR)

        left_up_point = (x - r, y - r)
        right_down_point = (x + r, y + r)
        two_point_list = [left_up_point, right_down_point]

        draw.ellipse(two_point_list, fill=color)

        if i > 0:

            info_prev = info_about_photo[len(info_about_photo) - i]

            x_prev = info_prev[0] + diff_r
            y_prev = info_prev[1] + diff_r

            points = find_magic_points(x, y, x_prev, y_prev, r)

            if points is not None:
                draw.polygon(points, fill=color)

    np_img = np.array(image)

    return scale_image(np_img, img_size, r)


def scale_image(np_img, img_size, r):

    non_white_indices = np.where(np_img < 255)
    row_indices, column_indices = non_white_indices[0], non_white_indices[1]

    if len(row_indices) == 0:
        return None

    top = np.min(row_indices)
    bottom = np.max(row_indices)
    left = np.min(column_indices)
    right = np.max(column_indices)

    diff_y = bottom - top
    diff_x = right - left

    final_size = max(diff_y, diff_x) + 1

    img_without_white_spaces = np_img[top:bottom+1, left:right+1]
    square_img = np.full((final_size, final_size), 255)

    if diff_y > diff_x:
        new_top = 0
        new_bottom = diff_y
        new_left = (diff_y - diff_x) // 2
        new_right = new_left + diff_x

    else:
        new_top = (diff_x - diff_y) // 2
        new_bottom = new_top + diff_y
        new_left = 0
        new_right = diff_x

    square_img[new_top:new_bottom+1, new_left:new_right+1] = img_without_white_spaces

    final_img = Image.fromarray(square_img).resize((Parameters.FINAL_SIZE, Parameters.FINAL_SIZE))

    return np.array(final_img)


def find_magic_points(x, y, x_prev, y_prev, r):

    x_length = x_prev - x
    y_length = y_prev - y
    diagonal = math.sqrt(x_length ** 2 + y_length ** 2)

    if diagonal < Parameters.MIN_LIMIT:
        return None

    sin_alfa = y_length / diagonal
    cos_alfa = x_length / diagonal

    p1 = (x_prev - r * sin_alfa, y_prev + r * cos_alfa)
    p2 = (x_prev + r * sin_alfa, y_prev - r * cos_alfa)
    p3 = (x + r * sin_alfa, y - r * cos_alfa)
    p4 = (x - r * sin_alfa, y + r * cos_alfa)

    return [p1, p2, p3, p4]

