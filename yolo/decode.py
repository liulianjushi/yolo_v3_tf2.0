import tensorflow as tf

from configuration import SCALE_SIZE, CATEGORY_NUM, IMAGE_HEIGHT, ANCHOR_NUM_EACH_SCALE
from yolo.anchor import get_coco_anchors
from yolo.nms import NMS


class Decode(tf.keras.Model):
    def __init__(self):
        super(Decode, self).__init__()
        self.nms = NMS()

    def __generate_grid_index(self, grid_dim):
        x = tf.range(grid_dim, dtype=tf.dtypes.float32)
        y = tf.range(grid_dim, dtype=tf.dtypes.float32)
        X, Y = tf.meshgrid(x, y)
        X = tf.reshape(X, shape=(-1, 1))
        Y = tf.reshape(Y, shape=(-1, 1))
        return tf.concat(values=[X, Y], axis=-1)

    def __bounding_box_predict(self, feature_map, scale_type, is_training=False):
        h = feature_map.shape[1]
        w = feature_map.shape[2]
        if h != w:
            raise ValueError("The shape[1] and shape[2] of feature map must be the same value.")
        area = h * w
        pred = tf.reshape(feature_map, shape=(-1, ANCHOR_NUM_EACH_SCALE * area, CATEGORY_NUM + 5))
        # pred = tf.nn.sigmoid(pred)
        tx_ty, tw_th, confidence, class_prob = tf.split(pred, num_or_size_splits=[2, 2, 1, CATEGORY_NUM], axis=-1)
        confidence = tf.nn.sigmoid(confidence)
        class_prob = tf.nn.sigmoid(class_prob)
        center_index = self.__generate_grid_index(grid_dim=h)
        center_index = tf.tile(center_index, [1, ANCHOR_NUM_EACH_SCALE])
        center_index = tf.reshape(center_index, shape=(1, -1, 2))
        # shape : (1, 507, 2), (1, 2028, 2), (1, 8112, 2)

        center_coord = center_index + tf.nn.sigmoid(tx_ty)
        anchors = tf.tile(get_coco_anchors(scale_type) / IMAGE_HEIGHT,
                          [area, 1])  # shape: (507, 2), (2028, 2), (8112, 2)
        bw_bh = tf.math.exp(tw_th) * anchors

        box_xy = center_coord / h
        box_wh = bw_bh

        # reshape
        center_index = tf.reshape(center_index, shape=(-1, h, w, ANCHOR_NUM_EACH_SCALE, 2))
        box_xy = tf.reshape(box_xy, shape=(-1, h, w, ANCHOR_NUM_EACH_SCALE, 2))
        box_wh = tf.reshape(box_wh, shape=(-1, h, w, ANCHOR_NUM_EACH_SCALE, 2))
        feature_map = tf.reshape(feature_map, shape=(-1, h, w, ANCHOR_NUM_EACH_SCALE, CATEGORY_NUM + 5))

        # cast dtype
        center_index = tf.cast(center_index, dtype=tf.dtypes.float32)
        box_xy = tf.cast(box_xy, dtype=tf.dtypes.float32)
        box_wh = tf.cast(box_wh, dtype=tf.dtypes.float32)

        if is_training:
            return box_xy, box_wh, center_index, feature_map
        else:
            return box_xy, box_wh, confidence, class_prob

    @tf.function
    def __yolo_post_processing(self, feature, scale_type):
        box_xy, box_wh, confidence, class_prob = self.__bounding_box_predict(feature_map=feature,
                                                                             scale_type=scale_type,
                                                                             is_training=False)
        min_xy = box_xy - box_wh / 2
        max_xy = box_xy + box_wh / 2
        boxes = tf.concat(values=[min_xy, max_xy], axis=-1)
        boxes = tf.reshape(boxes, shape=(-1, 4))
        box_scores = confidence * class_prob
        box_scores = tf.reshape(box_scores, shape=(-1, CATEGORY_NUM))
        return boxes, box_scores

    def call(self, inputs, **kwargs):
        boxes_list = []
        box_scores_list = []
        for i in range(len(SCALE_SIZE)):
            boxes, box_scores = self.__yolo_post_processing(feature=inputs[i],
                                                            scale_type=i)
            boxes_list.append(boxes)
            box_scores_list.append(box_scores)
        boxes_array = tf.concat(boxes_list, axis=0)
        box_scores_array = tf.concat(box_scores_list, axis=0)
        box_array, score_array, class_array = self.nms([boxes_array, box_scores_array])
        return [box_array, score_array, class_array]


if __name__ == '__main__':
    decode = Decode()
    decode.build(input_shape=[(None, 13, 13, 255), (None, 26, 26, 255), (None, 52, 52, 255)])
    decode.summary()
