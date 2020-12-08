import tensorflow as tf


class NMS(tf.keras.layers.Layer):
    def __init__(self, max_box_num=20, category_num=20, confidence_threshold=0.5, iou_threshold=0.5):
        super(NMS, self).__init__()
        self.max_box_num = max_box_num
        self.num_class = category_num
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def call(self, inputs, **kwargs):
        boxes, box_scores = inputs
        mask = box_scores >= self.confidence_threshold
        box_list = []
        score_list = []
        class_list = []
        for i in range(self.num_class):
            box_of_class = tf.boolean_mask(boxes, mask[:, i])
            score_of_class = tf.boolean_mask(box_scores[:, i], mask[:, i])
            selected_indices = tf.image.non_max_suppression(boxes=box_of_class,
                                                            scores=score_of_class,
                                                            max_output_size=self.max_box_num,
                                                            iou_threshold=self.iou_threshold)
            selected_boxes = tf.gather(box_of_class, selected_indices)
            selected_scores = tf.gather(score_of_class, selected_indices)
            classes = tf.ones_like(selected_scores, dtype=tf.dtypes.int32) * i
            box_list.append(selected_boxes)
            score_list.append(selected_scores)
            class_list.append(classes)
        box_array = tf.concat(values=box_list, axis=0)
        score_array = tf.concat(values=score_list, axis=0)
        class_array = tf.concat(values=class_list, axis=0)

        return box_array, score_array, class_array
