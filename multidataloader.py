# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, mode):
        self.data_path = data_path
        self.params = params
        self.mode = mode

        self.left_image_batch = None
        self.right_image_batch = None
        self.right1_image_batch=None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values


        left_image_path = tf.string_join([self.data_path, split_line[0]])
        right_image_path = tf.string_join([self.data_path, split_line[1]])
        right1_image_path = tf.string_join([self.data_path, split_line[2]])

        left_image_o = self.read_image(left_image_path)
        right_image_o=self.read_image(right_image_path)
        right1_image_o = self.read_image(right1_image_path)

        if mode == 'train':

            do_augment = tf.random_uniform([], 0, 1)
            left_image, right_image,right1_image = tf.cond(do_augment > 0.5,
                                              lambda: self.augment_image_pair(left_image_o, right_image_o,right1_image_o),
                                              lambda: (left_image_o, right_image_o,right1_image_o))


            left_image.set_shape([None, None, 3])
            right_image.set_shape([None, None, 3])
            right1_image.set_shape([None, None, 3])

            min_after_dequeue = 48
            capacity = min_after_dequeue + 2 * params.batch_size
            self.left_image_batch, self.right_image_batch,self.right1_image_batch = tf.train.shuffle_batch([left_image, right_image,right1_image],
                                                                                   params.batch_size, capacity,
                                                                                   min_after_dequeue,
                                                                                   params.num_threads)

        else:
            return None

    def augment_image_pair(self, left_image, right_image,centre_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug = left_image ** random_gamma
        centre_image_aug=centre_image ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug = left_image_aug * random_brightness
        centre_image_aug = centre_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_image_aug *= color_image
        centre_image_aug *=color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
        centre_image_aug=tf.clip_by_value(centre_image_aug, 0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug,centre_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                        lambda: tf.image.decode_png(tf.read_file(image_path)))


        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
