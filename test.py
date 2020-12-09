import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from imutils.paths import list_files

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, PASCAL_VOC_CLASSES
from utils.visualization_utils import draw_bounding_box_on_image, SELECTED_COLORS


def get_transform_coefficient(w, h):
    if h <= w:
        longer_edge = "w"
        scale = IMAGE_WIDTH / w
        padding_length = (IMAGE_HEIGHT - h * scale) / 2
    else:
        longer_edge = "h"
        scale = IMAGE_HEIGHT / h
        padding_length = (IMAGE_WIDTH - w * scale) / 2
    return longer_edge, scale, padding_length


def pre_process_image(image):
    image = np.array(image)
    image_tensor = tf.image.resize_with_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image_tensor = tf.cast(image_tensor, tf.float32)
    image_tensor = image_tensor / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor


def post_process_image(boxes, image_shape):
    w, h = image_shape
    longer_edge, scale, padding_length = get_transform_coefficient(w, h)
    boxes = boxes * [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]
    if longer_edge == "h":
        boxes -= [padding_length, 0, padding_length, 0]
    else:
        boxes -= [0, padding_length, 0, padding_length]
    return boxes / scale


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.list_physical_devices(device_type="GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # load model
    yolo_v3 = tf.saved_model.load("saved_model/final")

    files_list = list_files("test_data", validExts="jpg")
    for file in files_list:
        image = Image.open(file)
        img_tensor = pre_process_image(image)
        img_tensor = img_tensor / 255.0
        boxes, scores, classes = yolo_v3(img_tensor, training=False)
        final_boxes = post_process_image(boxes, image.size)
        for box, scs, cls in zip(final_boxes.numpy(), scores.numpy(), classes.numpy()):
            display_str = f"{list(PASCAL_VOC_CLASSES.keys())[list(PASCAL_VOC_CLASSES.values()).index(cls + 1)]}:{tf.round(100 * scs)}%"
            draw_bounding_box_on_image(image, box[1], box[0], box[3], box[2], color=SELECTED_COLORS[cls],
                                       display_str=display_str)
        plt.imshow(image)
        plt.show()
