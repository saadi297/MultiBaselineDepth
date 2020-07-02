# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math as m

from bilinear_sampler import *

monodepth_parameters = namedtuple('parameters',
                                  'encoder, '
                                  'height, width, '
                                  'batch_size, '
                                  'num_threads, '
                                  'num_epochs, '
                                  'use_deconv, '
                                  'alpha_image_loss, '
                                  'disp_gradient_loss_weight, '
                                  'scaling_factor, '
                                  'full_summary')


class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, left, right, right1, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.right1 = right1
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)


    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1,  'SAME')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1,  'SAME') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1,  'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1,  'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        disp = 0.08*self.conv(x, 1, 3, 1, tf.nn.sigmoid)
        return disp

    def get_disp_1(self, x):
        disp = 0.3*self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def get_disp_2(self, x):
        disp = 0.3 * self.conv(x, 1, 3, 1, tf.nn.sigmoid)
        return disp


    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def build_vgg(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input, 32, 7)  # H/2
            conv2 = self.conv_block(conv1, 64, 5)  # H/4
            conv3 = self.conv_block(conv2, 128, 3)  # H/8
            conv4 = self.conv_block(conv3, 256, 3)  # H/16
            conv5 = self.conv_block(conv4, 512, 3)  # H/32
            conv6 = self.conv_block(conv5, 512, 3)  # H/64
            conv7 = self.conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)


        with tf.variable_scope('decoder_1'):
            upconv7_1 = upconv(conv7, 512, 3, 2)  # H/64
            concat7_1 = tf.concat([upconv7_1, skip6], 3)
            iconv7_1 = conv(concat7_1, 512, 3, 1)


            upconv6_1 = upconv(iconv7_1, 512, 3, 2)  # H/32
            concat6_1 = tf.concat([upconv6_1, skip5], 3)
            iconv6_1 = conv(concat6_1, 512, 3, 1)

            upconv5_1 = upconv(iconv6_1, 256, 3, 2)  # H/16
            concat5_1 = tf.concat([upconv5_1, skip4], 3)
            iconv5_1 = conv(concat5_1, 256, 3, 1)

            upconv4_1 = upconv(iconv5_1, 128, 3, 2)  # H/8
            concat4_1 = tf.concat([upconv4_1, skip3], 3)
            iconv4_1 = conv(concat4_1, 128, 3, 1)
            self.disp4_1 = self.get_disp_1(iconv4_1)
            udisp4_1 = self.upsample_nn(self.disp4_1, 2)

            upconv3_1 = upconv(iconv4_1, 64, 3, 2)  # H/4
            concat3_1 = tf.concat([upconv3_1, skip2, udisp4_1], 3)
            iconv3_1 = conv(concat3_1, 64, 3, 1)
            self.disp3_1 = self.get_disp_1(iconv3_1)
            udisp3_1 = self.upsample_nn(self.disp3_1, 2)

            upconv2_1 = upconv(iconv3_1, 32, 3, 2)  # H/2
            concat2_1 = tf.concat([upconv2_1, skip1, udisp3_1], 3)
            iconv2_1 = conv(concat2_1, 32, 3, 1)
            self.disp2_1 = self.get_disp_1(iconv2_1)
            udisp2_1 = self.upsample_nn(self.disp2_1, 2)

            upconv1_1 = upconv(iconv2_1, 16, 3, 2)  # H
            concat1_1 = tf.concat([upconv1_1, udisp2_1], 3)
            iconv1_1 = conv(concat1_1, 16, 3, 1)
            self.disp1_1 = self.get_disp_1(iconv1_1)

        with tf.variable_scope('decoder_2'):
            upconv7_2 = upconv(conv7, 512, 3, 2)  # H/64
            concat7_2 = tf.concat([upconv7_2, skip6], 3)
            iconv7_2 = conv(concat7_2, 512, 3, 1)

            upconv6_2 = upconv(iconv7_2, 512, 3, 2)  # H/32
            concat6_2 = tf.concat([upconv6_2, skip5], 3)
            iconv6_2 = conv(concat6_2, 512, 3, 1)

            upconv5_2 = upconv(iconv6_2, 256, 3, 2)  # H/16
            concat5_2 = tf.concat([upconv5_2, skip4], 3)
            iconv5_2 = conv(concat5_2, 256, 3, 1)

            upconv4_2 = upconv(iconv5_2, 128, 3, 2)  # H/8
            concat4_2 = tf.concat([upconv4_2, skip3], 3)
            iconv4_2 = conv(concat4_2, 128, 3, 1)
            self.disp4_2 = self.get_disp_2(iconv4_2)
            udisp4_2 = self.upsample_nn(self.disp4_2, 2)

            upconv3_2 = upconv(iconv4_2, 64, 3, 2)  # H/4
            concat3_2 = tf.concat([upconv3_2, skip2, udisp4_2], 3)
            iconv3_2 = conv(concat3_2, 64, 3, 1)
            self.disp3_2 = self.get_disp_2(iconv3_2)
            udisp3_2 = self.upsample_nn(self.disp3_2, 2)

            upconv2_2 = upconv(iconv3_2, 32, 3, 2)  # H/2
            concat2_2 = tf.concat([upconv2_2, skip1, udisp3_2], 3)
            iconv2_2 = conv(concat2_2, 32, 3, 1)
            self.disp2_2 = self.get_disp_2(iconv2_2)
            udisp2_2 = self.upsample_nn(self.disp2_2, 2)

            upconv1_2 = upconv(iconv2_2, 16, 3, 2)  # H
            concat1_2 = tf.concat([upconv1_2, udisp2_2], 3)
            iconv1_2 = conv(concat1_2, 16, 3, 1)
            self.disp1_2 = self.get_disp_2(iconv1_2)


    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid = self.scale_pyramid(self.left, 4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                    self.right1_pyramid = self.scale_pyramid(self.right1, 4)


                self.model_input = self.left

                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                else:
                    return None


    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_est_new = [self.disp1_1, self.disp2_1, self.disp3_1, self.disp4_1]
            self.disp_est_new_1 = [self.disp1_2, self.disp2_2, self.disp3_2, self.disp4_2]

            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]

            self.disp_left1_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est_new]
            self.disp_right1_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est_new]

            self.disp_left2_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est_new_1]


            if self.mode == 'test':
                return





        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i]) for i in range(4)]

            self.left1_est = [self.generate_image_left(self.right1_pyramid[i], self.disp_left1_est[i]) for i in range(4)]
            self.right1_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right1_est[i]) for i in range(4)]

            self.left2_est = [self.generate_image_left(self.right1_pyramid[i], self.disp_left2_est[i]) for i in range(4)]



        with tf.variable_scope('left-right'):


            self.right1_to_left1_disp = [self.generate_image_left(self.disp_right1_est[i], self.disp_left1_est[i]) for i in
                                         range(4)]
            self.left1_to_right1_disp = [self.generate_image_right(self.disp_left1_est[i], self.disp_right1_est[i]) for i in
                                        range(4)]

            self.forward = [tf.abs(512 * self.right1_to_left1_disp[i] - 512 * self.disp_left1_est[i]) for i in range(4)]
            self.forward_mask = [tf.to_float(tf.math.greater(self.forward[i], 1)) for i in range(4)]



        with tf.variable_scope('smoothness'):

            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_pyramid)

            self.disp_left1_smoothness = self.get_disparity_smoothness(self.disp_left1_est, self.left_pyramid)
            self.disp_right1_smoothness = self.get_disparity_smoothness(self.disp_right1_est, self.right1_pyramid)

            self.disp_left2_smoothness = self.get_disparity_smoothness(self.disp_left2_est, self.left_pyramid)



    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):

            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]


            self.l1_left1 = [tf.abs(self.left1_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_left2 = [(1-tf.to_float(self.forward_mask[i]))*tf.abs(self.left2_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left1 = [tf.reduce_mean(l) for l in self.l1_left1]
            self.l1_reconstruction_loss_left2 = [tf.reduce_mean(l) for l in self.l1_left2]
            self.l1_right1 = [tf.abs(self.right1_est[i] - self.right1_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right1 = [tf.reduce_mean(l) for l in self.l1_right1]


            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]


            self.ssim_left1 = [self.SSIM(self.left1_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_left2 = [(1-tf.to_float(self.forward_mask[i]))*self.SSIM(self.left2_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left1 = [tf.reduce_mean(s) for s in self.ssim_left1]
            self.ssim_loss_left2 = [tf.reduce_mean(s) for s in self.ssim_left2]
            self.ssim_right1 = [self.SSIM(self.right1_est[i], self.right1_pyramid[i]) for i in range(4)]
            self.ssim_loss_right1 = [tf.reduce_mean(s) for s in self.ssim_right1]


            # WEIGTHED SUM
            self.image_loss_left = [
                self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left[i] for i in range(4)]


            self.image_loss_left1 = [
                self.params.alpha_image_loss * self.ssim_loss_left1[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left1[i] for i in range(4)]
            self.image_loss_right1 = [
                self.params.alpha_image_loss * self.ssim_loss_right1[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_right1[i] for i in range(4)]

            self.image_loss_left2 = [
                self.params.alpha_image_loss * self.ssim_loss_left2[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left2[i] for i in range(4)]


            self.left_occlusion_l1=[self.forward_mask[i]*tf.abs(self.disp_left2_est[i] - self.params.scaling_factor*tf.stop_gradient(self.disp_left_est[i])) for i in range(4)]
            self.left_occlusion_l1_loss=[tf.reduce_mean(s) for s in self.left_occlusion_l1]


            self.left_occlusion_ssim = [self.forward_mask[i]*self.SSIM(self.disp_left2_est[i], self.params.scaling_factor*tf.stop_gradient(self.disp_left_est[i])) for i in range(4)]
            self.left_occlusion_ssim_loss = [tf.reduce_mean(s) for s in self.left_occlusion_ssim]


            self.left_occlusion_loss = [
                self.params.alpha_image_loss * self.left_occlusion_ssim_loss[i] + (1 - self.params.alpha_image_loss) *
                self.left_occlusion_l1_loss[i] for i in range(4)]

            self.occlusion_loss=tf.add_n(self.left_occlusion_loss)


            self.image_loss = tf.add_n(self.image_loss_left)
            self.image_loss_1 = tf.add_n(self.image_loss_left1 + self.image_loss_right1)
            self.image_loss_2=tf.add_n(self.image_loss_left2)



            # DISPARITY SMOOTHNESS
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]


            self.disp_left1_loss = [tf.reduce_mean(tf.abs(self.disp_left1_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_right1_loss = [tf.reduce_mean(tf.abs(self.disp_right1_smoothness[i])) / 2 ** i for i in range(4)]

            self.disp_left2_loss = [tf.reduce_mean(tf.abs(self.disp_left2_smoothness[i])) / 2 ** i for i in range(4)]


            self.disp_gradient_loss = tf.add_n(self.disp_left1_loss + self.disp_right1_loss +self.disp_left_loss  +self.disp_left2_loss)


            # TOTAL LOSS
            self.total_loss =self.image_loss+self.image_loss_1+self.image_loss_2+ self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.occlusion_loss



    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):

                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i],
                                  collections=self.model_collection)

                tf.summary.scalar('image_loss1_' + str(i), self.image_loss_left1[i] + self.image_loss_right1[i],
                                  collections=self.model_collection)

                tf.summary.scalar('image_loss2_' + str(i), self.image_loss_left2[i],
                                  collections=self.model_collection)

                tf.summary.scalar('occlusion_loss_' + str(i), self.left_occlusion_loss[i],
                                  collections=self.model_collection)

                tf.summary.image('disp_left_est_' + str(i),self.disp_left_est[i], max_outputs=4,
                                 collections=self.model_collection)


                tf.summary.image('disp_left_est1_' + str(i), self.disp_left1_est[i], max_outputs=4,
                                 collections=self.model_collection)
                tf.summary.image('disp_right_est1_' + str(i), self.disp_right1_est[i], max_outputs=4,
                                 collections=self.model_collection)

                tf.summary.image('disp_left_est2_' + str(i), self.disp_left2_est[i], max_outputs=4,
                                 collections=self.model_collection)

                tf.summary.image('forward_mask_' + str(i), tf.to_float(self.forward_mask[i]), max_outputs=4,
                                 collections=self.model_collection)



                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)

                    tf.summary.image('left1_est_' + str(i), self.left1_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('right1_est_' + str(i), self.right1_est[i], max_outputs=4,
                                     collections=self.model_collection)


                    tf.summary.image('left2_est_' + str(i), self.left2_est[i], max_outputs=4,
                                     collections=self.model_collection)


            if self.params.full_summary:
                tf.summary.image('left', self.left, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right1', self.right1, max_outputs=4, collections=self.model_collection)

