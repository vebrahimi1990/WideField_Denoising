import numpy as np
import tensorflow as tf
from keras.models import Input, Model

patch_size = 128
mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')
vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                  input_tensor=Input(shape=(patch_size, patch_size, 3)))

inter_vgg = []
LL = [1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17]
# LL = [1, 2, 4, 5, 7]
for i in LL:
    inter_vgg.append(Model(inputs=vgg.input, outputs=vgg.get_layer(vgg.layers[i].name).output))

mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')


def norm_mse_loss(prediction, gt):
    n_mse = mse(prediction, gt)
    norm_mse = tf.squeeze(n_mse)
    return norm_mse


# def norm_mse_loss(pred, gt):
#     mse1 = tf.keras.metrics.mean_squared_error(pred, gt)
#     mse1 = tf.math.reduce_sum(mse1, axis=(1, 2))
#     norm = tf.norm(gt, axis=(1, 2))
#     norm = tf.squeeze(norm)
#     norm = tf.pow(norm, 2)
#     norm = tf.math.reduce_sum(norm)
#     nmse = tf.math.divide(mse1, norm)
#     nmse = tf.math.reduce_mean(nmse)
#     return nmse

def ch_loss(pred, gt):
    norm = tf.norm(pred - gt, axis=(1, 2))
    norm = tf.squeeze(norm)
    norm = tf.pow(norm, 2)
    norm = norm/(256*256) + 1e-6
    norm = tf.pow(norm, 0.5)
    c_loss = tf.math.reduce_mean(norm)
    return c_loss


def edge_loss(pred, gt):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel = kernel.reshape((3, 3, 1, 1))
    pred = tf.nn.conv2d(pred, kernel, strides=[1, 1, 1, 1], padding='VALID')
    gt = tf.nn.conv2d(gt, kernel, strides=[1, 1, 1, 1], padding='VALID')
    e_loss = ch_loss(pred, gt)
    return e_loss


def fft_loss(prediction, gt):
    prediction = tf.transpose(prediction, perm=[0, 3, 1, 2])
    gt = tf.transpose(gt, perm=[0, 3, 1, 2])

    fft_prediction = tf.signal.fftshift(tf.signal.rfft2d(prediction))
    fft_gt = tf.signal.fftshift(tf.signal.rfft2d(gt))

    fft_prediction = tf.transpose(fft_prediction, perm=[0, 2, 3, 1])
    fft_gt = tf.transpose(fft_gt, perm=[0, 2, 3, 1])

    loss = norm_mse_loss(abs(fft_prediction), abs(fft_gt))
    loss = tf.cast(loss, tf.float32)
    return loss


def ssim_loss(prediction, gt):
    loss = 1.0 - tf.math.reduce_mean(tf.image.ssim(prediction, gt, max_val=1))
    return loss


def perceptual_loss(prediction, gt):
    loss = 0
    prediction = tf.image.grayscale_to_rgb(prediction)
    gt = tf.image.grayscale_to_rgb(gt)
    for m in range(len(LL)):
        vgg_prediction = inter_vgg[m](prediction)
        vgg_gt = inter_vgg[m](gt)
        loss = loss + norm_mse_loss(vgg_prediction, vgg_gt)
    loss = loss / len(LL)
    return loss


# def generator_loss(prediction, gt):
#     percept_loss = perceptual_loss(prediction, gt)
#     # ft_loss = fft_loss(prediction, gt)
#     norm_mse = norm_mse_loss(prediction, gt)
#     sim_loss = ssim_loss(prediction, gt)
#     gen_loss = 0.16 * norm_mse + 0.84 * sim_loss + 1e-3 * percept_loss
#     # gen_loss = norm_mse
#     return gen_loss

def generator_loss(prediction, gt):
    c_loss = ch_loss(prediction, gt)
    e_loss = edge_loss(prediction, gt)
    gen_loss = c_loss + 0.05 * e_loss
    return gen_loss
