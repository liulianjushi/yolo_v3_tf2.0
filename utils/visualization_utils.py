import io
import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont, Image

from configuration import PASCAL_VOC_CLASSES

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

SELECTED_COLORS = random.sample(STANDARD_COLORS, len(PASCAL_VOC_CLASSES))


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=4, display_str=""):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    # Each display_str has a top and bottom margin of 0.05x.
    text_width, text_height = font.getsize(display_str)
    if top > text_height:
        text_bottom = top
    else:
        text_bottom = bottom + text_height
    # Reverse list and print from bottom to top.

    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill='black',
              font=font)
    text_bottom -= text_height - 2 * margin


def display_image(images, labels):
    imgs = []
    for index, (img, height, width) in enumerate(zip(images["raw"], images["height"], images["width"])):
        # mask = tf.not_equal(label[..., -1], 0)
        # boxes = tf.boolean_mask(label[..., :4], mask=mask, axis=0)
        boxes = labels[8692:8728]
        # boxes = tf.concat([boxes[..., 0:2] - boxes[..., 2:4] / 2, boxes[..., 0:2] + boxes[..., 2:4] / 2], axis=-1)
        boxes = tf.gather(boxes, [1, 0, 3, 2], axis=1)
        img = tf.expand_dims(img + 0.5, 0)
        colors = np.array([[1.0, 0.0, 0.0]])
        img = tf.image.draw_bounding_boxes(img, tf.expand_dims(boxes, 0), colors)
        imgs.append(tf.squeeze(img, axis=0))
    return imgs





def plot_to_image(images, labels, pre_labels=None):
    imgs = []

    for index, (img, height, width, label) in enumerate(zip(images["raw"], images["height"], images["width"], labels)):
        figure = plt.figure()
        ax = plt.Axes(figure, [0., 0., 1., 1.])
        # 关闭子图的坐标轴
        ax.set_axis_off()
        figure.add_axes(ax)
        img = tf.image.resize(img, (height, width))
        img = tf.cast((img + 0.5) * 255, tf.uint8)
        if pre_labels is not None:
            img = tf.concat([img, img], 1)
            img_pil = Image.fromarray(img.numpy())
            predict_boxes = pre_labels["detection_boxes"][index]
            detection_classes = pre_labels["detection_classes"][index]
            detection_scores = pre_labels["detection_scores"][index]
            detection_num = pre_labels["detection_num"][index]
            print(f"detection_classes:{detection_classes.numpy().tolist()}")
            if detection_num != 0:
                for i, box in enumerate(predict_boxes):
                    box = tf.cast(box * [width, height, width, height] + [width, 0, width, 0], tf.int32)
                    draw_bounding_box_on_image(img_pil, box[1], box[0], box[3], box[2],
                                               color=SELECTED_COLORS[detection_classes[i] - 1],
                                               thickness=4,
                                               display_str=f"{OBJECT_CLASSES[detection_classes[i] - 1]}:{tf.round(100 * detection_scores[i])}%")
        else:
            img_pil = Image.fromarray(img.numpy())
        mask = tf.not_equal(label[..., -1], 0)
        boxes = tf.cast(tf.boolean_mask(label, mask=mask, axis=0) * [width, height, width, height, 1], tf.int32)
        for box in boxes:
            draw_bounding_box_on_image(img_pil, box[1], box[0], box[3], box[2], color=SELECTED_COLORS[box[4] - 1],
                                       thickness=4, display_str=OBJECT_CLASSES[box[4] - 1])
        plt.imshow(img_pil)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        imgs.append(image)
    return imgs
