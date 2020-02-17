from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
import json
import glob
import random
import collections
import math
import time
from PIL import Image, ImageTk
import FP_matcher
import pickle
import io
import tkinter
from tkinter import messagebox, font
from tkinter import filedialog, ttk
from ctypes import *

# for liveness detection
from keras import backend as K
from scipy.misc import imread, imsave, imresize
import preModels.new_gram as gram

d_fp_parser = argparse.ArgumentParser()
d_fp_parser.add_argument("--which_direction", type=str, default="BtoA", choices=["AtoB", "BtoA"])            #default
d_fp_parser.add_argument("--input_dir", default="tmp", help="path to folder containing images")    #default
d_fp_parser.add_argument("--mode", default="test", choices=["train", "test", "export"])                     #default
d_fp_parser.add_argument("--output_dir", default="generated_fp", help="where to put output files")                #default
d_fp_parser.add_argument("--max_epochs", type=int, default=200, help="number of training epochs")            #default = 200
d_fp_parser.add_argument("--seed", default=189649830, type=int)
d_fp_parser.add_argument("--checkpoint", default="checkpoint", help="directory with checkpoint to resume training from or use for testing")
d_fp_parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
d_fp_parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
d_fp_parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
d_fp_parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
d_fp_parser.add_argument("--display_freq", type=int, default=1000, help="write current training images every display_freq steps")
d_fp_parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
d_fp_parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
d_fp_parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
d_fp_parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
d_fp_parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
d_fp_parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
d_fp_parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
d_fp_parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
d_fp_parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
d_fp_parser.add_argument("--aspect_ratio", type=float, default=2.0, help="aspect ratio of output images (width/height)")
# export options
d_fp_parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = d_fp_parser.parse_args()

targetHeight = 320
targetWidth = 320
imgHeight = 320
imgWidth = 280
pH = 0
pW = 20

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs")


def reading_from_sensor(num_fp):
    myDll = cdll.LoadLibrary('izzixOEMAPI32_x64.dll')

    MAX_SIZE = 400*302

    RawImage = c_ubyte * MAX_SIZE
    Feature = c_ubyte * 1024

    pRawImage = RawImage()
    pFeature = Feature()

    result = myDll.GetCaptureImageEx(0, byref(pRawImage), 100, 40)
    # result = myDll.GetFinger(0, byref(pRawImage), byref(pFeature))
    f = open('foo{}.raw'.format(num_fp), 'wb')
    f.write((''.join(chr(i) for i in pRawImage)).encode('charmap'))

    rawImage = open('foo{}.raw'.format(num_fp), 'rb').read()
    image = Image.frombytes('L', (280, 320), rawImage)
    image.save('foo{}.png'.format(num_fp))

    # image.show()

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.bmp"))
        decode = tf.image.decode_bmp

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([targetHeight, targetWidth, 3])

        # break apart image pair and move to range [-1, 1]
        a_images = preprocess(raw_input)
        b_images = preprocess(raw_input)

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(Aimg, Bimg):
        rA = Aimg
        rB = Bimg
        return rA, rB

    with tf.name_scope("transform_images"):
        target_images, input_images = transform(targets, inputs)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 16, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (a.ngf * 16, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    return Model(outputs=outputs,)

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    image_dir = a.output_dir
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        # for kind in ["inputs", "outputs", "targets"]:
        for kind in ["inputs", "outputs"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
            if (kind == 'outputs'):
                outputs_pil = Image.open(out_path)
    return outputs_pil

# create minutia map : start
def minutia_map_creation(minutia_set, height, width, target_size, fn):
    bs = 5
    px = int((target_size - width) / 2)
    py = int((target_size - height) / 2)
    map_1 = np.zeros([target_size, target_size], dtype=np.uint8)
    map_2 = np.zeros([target_size, target_size], dtype=np.uint8)
    # create two channels
    for minutia in minutia_set:
        x = minutia[0] + px
        y = minutia[1] + py
        theta = int(math.floor(minutia[2]/2 + 1))
        quality = minutia[3]
        type = minutia[4]
        if quality >= 20:
            i1 = max(0, y - bs)
            i2 = min(y + bs, target_size)
            j1 = max(0, x - bs)
            j2 = min(x + bs, target_size)
            map_1[i1:i2, j1:j2] = theta
            map_2[i1:i2, j1:j2] = type
    # stack two channels into one image
    map = np.stack((map_1, map_1, map_2), axis=-1)
    minutia_map = Image.fromarray(map)
    minutia_map.save(os.path.join(a.input_dir, fn),'png')
    # create minutia map : end

def d_fp_generation():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    # print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1:
            # crop from [targetHeight, targetWidth] to [imgHeight, imgWidth]
            image = tf.image.crop_to_bounding_box(image, pH, pW, imgHeight, imgWidth)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None

    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                outputs_pil = save_images(results)

            return outputs_pil, (time.time() - start)/max_steps

def master_minutia_generation(seed_value):
    # Initialize TensorFlow session.
    tf.reset_default_graph()
    with tf.name_scope("master_minutiae_generation"):
        sess_ps_gan = tf.InteractiveSession()
        with open('network-snapshot-012000.pkl', 'rb') as file:
            G, D, Gs = pickle.load(file)

        start = time.time()
        # Generate latent vectors.
        latents_10000 = np.random.RandomState(10000).randn(10000, *Gs.input_shapes[0][1:])  # 10000 random latents
        # hand-pick one vector

        # seed_value = input("Select a random seed between 0 to 9999: ")
        # seed_value = np.random.randint(0, 9999, size=1)
        # seed_value = np.random.RandomState().randint(0, 9999, size=1)
        print(seed_value)
        latents = latents_10000[[int(seed_value)]]

        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
        # images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

        sess_ps_gan.close()

        # Save images as PNG.
        t = images[0]
        t_arr = np.dstack((t, t, t))
        t_im = Image.fromarray(t_arr, 'RGB')
        t_im = t_im.resize((280, 320), Image.BICUBIC)
        # t_im.save('tmp/r_fp_' + str(seed_value) + '.png')

        imgByteArr = io.BytesIO()
        t_im.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr, (time.time() - start)

def global_rotation(min_set, offset, x0, y0):
    a = np.random.randint((-1)*offset, offset, size=1)
    ro = math.radians(a[0])
    # print(a[0])
    cos_t = math.cos(ro)
    sin_t = math.sin(ro)

    minutia_set_new = []
    for minutia in min_set:
        x = minutia[0] - x0
        y = minutia[1] - y0
        theta = minutia[2]
        quality = minutia[3]
        type = minutia[4]

        x_n = int(round(x*cos_t - y*sin_t) + x0)
        y_n = int(round(x*sin_t + y*cos_t) + y0)

        theta_n = theta + a[0]
        if theta_n > 360:
            theta_n = theta_n - 360
        else:
            if theta_n < 0:
                theta_n = theta_n + 360

        if x_n > 0 and x_n <= imgWidth and y_n > 0 and y_n <= imgHeight:
            minutia_set_new.append([x_n, y_n, theta_n, quality, type])
    return minutia_set_new

def global_transformation(min_set, offset):
    # a = np.random.RandomState().randint((-1)*offset, offset, size=2)
    a = np.random.RandomState().randint(offset-10, offset, size=2)
    tx = a[0]
    ty = a[1]
    print(tx)
    print(ty)

    minutia_set_new = []
    for minutia in min_set:
        x_n = minutia[0] + tx
        y_n = minutia[1] + ty
        theta = minutia[2]
        quality = minutia[3]
        type = minutia[4]
        if x_n > 0 and x_n <= imgWidth and y_n > 0 and y_n <= imgHeight:
            minutia_set_new.append([x_n, y_n, theta, quality, type])
    return minutia_set_new



save_image_1 = np.zeros((320, 280))
save_image_2 = np.zeros((320, 280))


def GUI():

    def save_image1():
        fileName = filedialog.asksaveasfile(title='Save image to disk', initialdir='_real_fingerprints',
                                            filetypes=(('Image files', '*.jpg *.png'), ('All files', '*.*')))
        if not fileName:
            print('Saving canceled')
            return
        print('This is file name from the first panel')
        print('file name: ', fileName.name)
        # save the first image to the selected location
        p = os.getcwd()
        save_image_1 = imread(p + '/temp1.jpg')
        imsave(fileName.name, save_image_1)


    def save_image2():
        fileName = filedialog.asksaveasfile(title='Save image to disk', initialdir='_real_fingerprints',
                                            filetypes=(('Image files', '*.jpg *.png'), ('All files', '*.*')))
        if not fileName:
            print('Saving canceled')
        print('This is file name from the second panel')
        print('file name: ', fileName.name)
        # save the second image to the selected location
        p = os.getcwd()
        save_image_2 = imread(p + '/temp2.jpg')
        imsave(fileName.name, save_image_2)

    def on_select(event=None):
        print('----------------------------')

        if event:  # <-- this works only with bind because `command=` doesn't send event
            if event.widget.get() == 'User selection':
                fn = filedialog.askopenfile(title='Select a pretrained model')
                if fn is not None:
                    str = fn.name + ' 1'
                else:
                    messagebox.showinfo('Info', 'No model was chosen, the default 2011_Biometrika will be selected')
                    str = '2011_Biometrika.hdf5 0'
            else:
                str = event.widget.get() + '.hdf5 0'
            print("event.widget:", str)
            f = open('info.txt', 'w')
            f.write(str)
            f.close()


        # for i, x in enumerate(all_comboboxes):
        #     print("all_comboboxes[%d]: %s" % (i, x.get()))

    def ExitApplication():
        MsgBox = tkinter.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application',
                                           icon='warning')
        if MsgBox == 'yes':
            rootWindow.destroy()
        else:
            panel_image_1.configure(image=ImageTk.PhotoImage(Image.open('logo_2.jpg')), text='')
            panel_image_2.configure(image=ImageTk.PhotoImage(Image.open('logo_2.jpg')), text='')
            str_score.set('Matching score: ')
            str_decision.set('Matching decision (at 0.01%FAR): ')
            str_ptime.set('Generation time: ')
            rootWindow.update()

    rootWindow = tkinter.Tk()
    rootWindow.geometry('675x580')
    rootWindow.wm_title('Fingerprint Generator - Liveness detection')

    s = ttk.Style()
    s.configure('my.TButton', font=('Helvetica', 12))
    popup1 = tkinter.Menu(rootWindow, tearoff=0)
    popup1.add_command(label='Save image', command=save_image1)
    popup1.add_separator()
    popup1.add_command(label='Exit', command=ExitApplication)

    popup2 = tkinter.Menu(rootWindow, tearoff=0)
    popup2.add_command(label='Save image', command=save_image2)
    popup2.add_separator()
    popup2.add_command(label='Exit', command=ExitApplication)

    def do_popup1(event):
        # display the popup menu
        try:
            popup1.tk_popup(event.x_root, event.y_root, 0)
        finally:
            popup1.grab_release()

    def do_popup2(event):
        # display the popup menu
        try:
            popup2.tk_popup(event.x_root, event.y_root, 0)
        finally:
            popup2.grab_release()


    def close_window():
        rootWindow.destroy()



    def openfn(str):
        # get the name of a file from the user, presenting her a dialog
        fileName = filedialog.askopenfilename(title=str, initialdir='_real_fingerprints',
                                              filetypes=(("Image files", "*.tif *.png *.jpg"), ("All files", "*.*")))
        return fileName


    def d_fp_generation_run_file():
        # select a real fingerprint and get the real image
        real_filename = openfn('Select a real fingerprint')

        if not real_filename:
            print('No image selected')
            return
        save_image_1 = np.array(Image.open(real_filename))
        imsave('temp1.jpg', save_image_1)
        d_fp_generation1(real_filename)

    def d_fp_generation_run_sensor():
        tkinter.messagebox.showinfo('Scan fingerprint', 'Please scan fingerprint')
        num_fp = 1
        reading_from_sensor(num_fp)
        p = os.getcwd()
        real_filename = p + '/foo{}.png'.format(num_fp)
        save_image_1 = imread(real_filename)
        imsave('temp1.jpg', save_image_1)
        d_fp_generation1(real_filename)

    def d_fp_generation1(real_filename):
        # clear tmp and generated_fp folder
        files = glob.glob(os.path.join(a.input_dir, '*.*'))
        for f in files:
            os.remove(f)
        files = glob.glob(os.path.join(a.output_dir, '*.*'))
        for f in files:
            os.remove(f)

        real_image = Image.open(real_filename)
        new_size = 280, 320
        real_image.thumbnail(new_size)

        # display the real fingerprint
        display_image = ImageTk.PhotoImage(real_image)
        panel_image_1.configure(image=display_image, text='real fingerprint')
        panel_image_1.image = display_image
        panel_image_1.text = 'real fingerprint'

        panel_image_2.configure(image=ImageTk.PhotoImage(Image.open('logo_2.jpg')), text='')
        str_score.set('Matching score: ')
        str_decision.set('Matching decision (at 0.01%FAR): ')
        str_ptime.set('Generation time: ')
        rootWindow.update()

        # extract minutiae
        img_width, img_height, img_quality, minutia_set = FP_matcher.SingleExtractFromFile(real_filename, False, '')
        minutia_map_creation(minutia_set, img_height, img_width, 320, '1.png')

        # generate d-fingerprint
        d_fp_image, ptime = d_fp_generation()
        imsave('temp2.jpg', np.array(d_fp_image))

        display_d_fp_image = ImageTk.PhotoImage(d_fp_image)
        panel_image_2.configure(image=display_d_fp_image, text='d-fingerprint')
        panel_image_2.image = display_d_fp_image
        panel_image_2.text = 'd-fingerprint'
        rootWindow.update()

        score, quality_real, quality_syn = FP_matcher.single_match_from_file(real_filename, os.path.join(a.output_dir,
                                                                                                         '1-outputs.png'))
        print(score)

        panel_image_1.configure(text='real fingerprint (Quality = ' + str(quality_real) + ')')
        panel_image_1.text = 'real fingerprint (Quality = ' + str(quality_real) + ')'
        panel_image_2.configure(text='d-fingerprint (Quality = ' + str(quality_syn) + ')')
        panel_image_2.text = 'd-fingerprint (Quality = ' + str(quality_syn) + ')'

        # display results
        str_score.set('Matching score: ' + str(score))
        if score >= 40:
            str_decision.set('Matching decision (at 0.01%FAR): Genuine')
        else:
            str_decision.set('Matching decision (at 0.01%FAR): Impostor')
        str_ptime.set('Generation time: ' + str(round(ptime, 2)) + ' s')
        tf.reset_default_graph()

    def d_fp_generation_run():
        MsgBox = tkinter.messagebox.askquestion('Select image', 'Do you want to capture fingerprint from sensor ?')
        if MsgBox == 'yes':
            d_fp_generation_run_sensor()
        else:
            d_fp_generation_run_file()


    def r_fp_generation_run():
        panel_image_1.configure(image=ImageTk.PhotoImage(Image.open('logo_2.jpg')), text='')
        panel_image_2.configure(image=ImageTk.PhotoImage(Image.open('logo_2.jpg')), text='')
        str_score.set('Matching score: ')
        str_decision.set('Matching decision (at 0.001%FAR): ')
        str_ptime.set('Generation time: ')
        rootWindow.update()
        # clear tmp and generated_fp folder
        files = glob.glob(os.path.join(a.input_dir, '*.*'))
        for f in files:
            os.remove(f)

        seed_value = np.random.RandomState().randint(0, 9999, size=1)
        r_fp_pattern, ptime_1 = master_minutia_generation(seed_value)
        tf.reset_default_graph()

        img_width, img_height, img_quality, minutia_set = FP_matcher.SingleExtractFromImage(r_fp_pattern)
        minutia_map_creation(minutia_set, img_height, img_width, 320, str(seed_value) + '_1.png')

        minutia_set_1 = global_rotation(minutia_set, 15, img_width/2, img_height/2)
        minutia_set_2 = global_transformation(minutia_set, 50)
        minutia_map_creation(minutia_set_2, img_height, img_width, 320, str(seed_value) + '_2.png')

        minutia_set_1 = global_rotation(minutia_set, 15, img_width / 2, img_height / 2)
        minutia_set_2 = global_transformation(minutia_set, 50)
        minutia_map_creation(minutia_set_2, img_height, img_width, 320, str(seed_value) + '_3.png')

        # generate d-fingerprint
        d_fp_image, ptime_2 = d_fp_generation()

        # matching
        r_fp_fn_1 = os.path.join(a.output_dir, str(seed_value) + '_1-outputs.png')
        r_fp_fn_2 = os.path.join(a.output_dir, str(seed_value) + '_2-outputs.png')
        r_fp_fn_3 = os.path.join(a.output_dir, str(seed_value) + '_3-outputs.png')
        score_1, img_quality_1, img_quality_2 = FP_matcher.single_match_from_file(r_fp_fn_1, r_fp_fn_2)
        score_2, img_quality_1, img_quality_3 = FP_matcher.single_match_from_file(r_fp_fn_1, r_fp_fn_3)
        score_3, _, _ = FP_matcher.single_match_from_file(r_fp_fn_2, r_fp_fn_3)
        max_score = max(score_1, score_2, score_3)
        if max_score == score_1:
            fn_1 = r_fp_fn_1
            fn_2 = r_fp_fn_2
            qscore_1 = img_quality_1
            qscore_2 = img_quality_2
        else:
            if max_score == score_2:
                fn_1 = r_fp_fn_1
                fn_2 = r_fp_fn_3
                qscore_1 = img_quality_1
                qscore_2 = img_quality_3
            else:
                fn_1 = r_fp_fn_2
                fn_2 = r_fp_fn_3
                qscore_1 = img_quality_2
                qscore_2 = img_quality_3
        print(max_score)

        # display the 1st generated fingerprints
        tmp_image = ImageTk.PhotoImage(Image.open(fn_1))
        panel_image_1.configure(image=tmp_image, text='r-fingerprint #1 (Quality = ' + str(qscore_1) + ')')
        panel_image_1.image = tmp_image
        panel_image_1.text = 'r-fingerprint #1 (Quality = ' + str(qscore_1) + ')'
        save_image_1 = np.array(Image.open(fn_1))
        imsave('temp1.jpg', save_image_1)

        # display the 2st generated fingerprints
        tmp_image = ImageTk.PhotoImage(Image.open(fn_2))
        panel_image_2.configure(image=tmp_image, text='r-fingerprint #2 (Quality = ' + str(qscore_2) + ')')
        panel_image_2.image = tmp_image
        panel_image_2.text = 'r-fingerprint #2 (Quality = ' + str(qscore_1) + ')'
        save_image_2 = np.array(Image.open(fn_2))
        imsave('temp2.jpg', save_image_2)

        # display results
        str_score.set('Matching score: ' + str(max_score))
        if max_score >= 60:
            str_decision.set('Matching decision (at 0.001%FAR): Genuine')
        else:
            str_decision.set('Matching decision (at 0.001%FAR): Impostor')
        str_ptime.set('Generation time: ' + str(round(ptime_1 + ptime_2, 2)) + ' s')
        tf.reset_default_graph()


    def livdet_run_file():
        # select a real fingerprint and get the real image
        real_filename = openfn('Select a real fingerprint')

        if not real_filename:
            print('No image selected')
            return
        save_image_1 = imread(real_filename)
        imsave('temp1.jpg', save_image_1)

        livdet(real_filename)

    def livdet_run_sensor():
        tkinter.messagebox.showinfo('Scan fingerprint', 'Please scan fingerprint')
        num_fp = 1
        reading_from_sensor(num_fp)
        real_filename = 'foo{}.png'.format(num_fp)

        p = os.getcwd()
        save_image_1 = imread(p + '/foo{}.png'.format(num_fp))
        imsave('temp1.jpg', save_image_1)
        livdet(real_filename)

    def livdet(real_filename):
        # read the name of classifier
        f = open('info.txt', 'r')
        st = f.readline().split(' ')
        str1 = st[0]
        u_sl = int(st[1])

        selected_image = Image.open(real_filename)
        w_ori, h_ori = selected_image.size

        new_size = 280, 280 * h_ori / w_ori
        resized_image = selected_image
        resized_image.thumbnail(new_size)

        # display the selected fingerprint
        # subs = real_filename.split('/')
        display_image = ImageTk.PhotoImage(resized_image)
        panel_image_1.configure(image=display_image, text='Groundtruth')
        panel_image_1.image = display_image
        panel_image_2.configure(image=ImageTk.PhotoImage(Image.open('logo_2.jpg')), text=' ')
        str_score.set('')
        str_decision.set('')
        str_ptime.set('')
        rootWindow.update()

        # liveness detection
        imgLoaded = cv2.imread(real_filename, cv2.IMREAD_GRAYSCALE) / 255.
        imgLoaded = cv2.resize(imgLoaded, dsize=None, fx=0.5, fy=0.5)
        input_shape = imgLoaded.shape

        if u_sl:
            fpadnet_model = gram.new_gram_models('gaj', input_shape, weights_path=str1)
            print('input_shape: {}'.format(input_shape))
            print('str1: {}'.format(str1))
        else:
            fpadnet_model = gram.new_gram_models('gaj', input_shape, weights_path='./models/' + str1)
        start = time.time()
        reShapeImage = imgLoaded.reshape(1, imgLoaded.shape[0], imgLoaded.shape[1], 1)
        features = fpadnet_model.predict(reShapeImage)
        print('features: {}'.format(features))
        predIndex = np.argmax(features)

        ptime = time.time() - start

        if predIndex == 1:
            str_score.set('Liveness detection result: ' + 'Fake fingerprint.')
        else:
            str_score.set('Liveness detection result: ' + 'Live fingerprint.')
        str_decision.set('Processing time: ' + str(round(ptime, 2)) + ' s')
        K.clear_session()


    def livdet_run():
        MsgBox = tkinter.messagebox.askquestion('Select source...', 'Do you want to capture fingerprint from sensor ?')
        if MsgBox == 'yes':
            livdet_run_sensor()
        else:
            livdet_run_file()

    def matching_run_sensor():
        # filename_1 = openfn('Select the first fingerprint')
        tkinter.messagebox.showinfo('Scan fingerprint', 'Please scan the first fingerprint')
        num_fp = 1
        reading_from_sensor(num_fp)
        filename_1 = 'foo{}.png'.format(num_fp)
        p = os.getcwd()
        save_image_1 = imread(p + '/foo1.png')
        imsave('temp1.jpg', save_image_1)

        tkinter.messagebox.showinfo('Scan fingerprint', 'Please scan the second fingerprint')
        num_fp = 2
        reading_from_sensor(num_fp)
        filename_2 = 'foo{}.png'.format(num_fp)
        save_image_2 = imread(p + '/foo2.png')
        imsave('temp2.jpg', save_image_2)
        matching(filename_1, filename_2)

    def matching_run_file():
        filename_1 = openfn('Select the first fingerprint')
        if not filename_1:
            print('No image selected')
            return
        save_image_1 = imread(filename_1)
        imsave('temp1.jpg', save_image_1)

        filename_2 = openfn('Select the second fingerprint')
        if not filename_2:
            print('No image selected')
            return
        save_image_2 = imread(filename_2)
        imsave('temp2.jpg', save_image_2)
        matching(filename_1, filename_2)

    # this function is used for the matching two fingerprint
    def matching_run():
        # read the first image: it can be from sensor or from the disk
        MsgBox = tkinter.messagebox.askquestion('Select the first image',
                                                'Do you want to capture fingerprint from sensor ?')
        if MsgBox == 'yes':
            tkinter.messagebox.showinfo('Scan the first fingerprint', 'Please scan the fingerprint')
            num_fp = 1
            reading_from_sensor(num_fp)
            filename_1 = 'foo{}.png'.format(num_fp)
            p = os.getcwd()
            save_image_1 = imread(p + '/foo1.png')

        else:
            filename_1 = openfn('Select the first fingerprint')
            if not filename_1:
                print('No image selected')
                return
            save_image_1 = imread(filename_1)
        imsave('temp1.jpg', save_image_1)

        # read the second image: it also can be from sensor or from the disk
        MsgBox = tkinter.messagebox.askquestion('Select the second image', 'Do you want to capture fingerprint from sensor ?')
        if MsgBox == 'yes':
            tkinter.messagebox.showinfo('Scan the second fingerprint', 'Please scan the fingerprint')
            num_fp = 2
            reading_from_sensor(num_fp)
            filename_2 = 'foo{}.png'.format(num_fp)
            p = os.getcwd()
            save_image_2 = imread(p + '/foo2.png')

        else:
            filename_2 = openfn('Select the second fingerprint')
            if not filename_2:
                print('No image selected')
                return
            save_image_2 = imread(filename_2)
        imsave('temp2.jpg', save_image_2)

        # do the matching between two loaded images
        matching(filename_1, filename_2)

    def matching(filename_1, filename_2):
        start = time.time()
        score, img_quality_1, img_quality_2 = FP_matcher.single_match_from_file(filename_1, filename_2)
        ptime = time.time() - start

        # display the 1st generated fingerprints
        tmp_image = ImageTk.PhotoImage(Image.open(filename_1))
        panel_image_1.configure(image=tmp_image, text='Fingerprint #1 (Quality = ' + str(img_quality_1) + ')')
        panel_image_1.image = tmp_image
        panel_image_1.text = 'Fingerprint #1 (Quality = ' + str(img_quality_1) + ')'

        # display the 2st generated fingerprints
        tmp_image = ImageTk.PhotoImage(Image.open(filename_2))
        panel_image_2.configure(image=tmp_image, text='Fingerprint #2 (Quality = ' + str(img_quality_2) + ')')
        panel_image_2.image = tmp_image
        panel_image_2.text = 'Fingerprint #2 (Quality = ' + str(img_quality_1) + ')'

        # display results
        str_score.set('Matching score: ' + str(score))
        if score >= 40:
            str_decision.set('Matching decision (at 0.01%FAR): Genuine')
        else:
            str_decision.set('Matching decision (at 0.01%FAR): Impostor')
        str_ptime.set('Matching time: ' + str(round(ptime, 2)) + ' s')



    # place for the first image
    tmp_image = ImageTk.PhotoImage(Image.open('logo_2.jpg'))
    panel_image_1 = ttk.Label(rootWindow, image=tmp_image, compound='bottom',
                           style='my.TButton', padding=5)
    panel_image_1.bind('<Button-3>', do_popup1)
    # print('name of file: ', save_image_name1)
    panel_image_1.image = tmp_image
    panel_image_1.place(x=20, y=15)

    # place for the second image
    panel_image_2 = ttk.Label(rootWindow, image=tmp_image, compound='bottom',
                               style='my.TButton', padding=5)
    panel_image_2.bind('<Button-3>', do_popup2)
    panel_image_2.image = tmp_image
    panel_image_2.place(x=355, y=15)

    # place for matching result
    tmp_image = ImageTk.PhotoImage(Image.open('logo_3.jpg'))
    place_for_result = ttk.Label(rootWindow, image=tmp_image, style='my.TButton')
    place_for_result.image = tmp_image
    place_for_result.place(x=20, y=440)

    # display results
    str_score = tkinter.StringVar()
    lbl_score = ttk.Label(rootWindow, textvariable=str_score, state='normal', font=("Helvetica", 12), borderwidth=0)
    str_score.set('Matching score: ')
    lbl_score.place(x=30, y=450)

    str_decision = tkinter.StringVar()
    lbl_decision = ttk.Label(rootWindow, textvariable=str_decision, state='normal', font=("Helvetica", 12), borderwidth=0)
    str_decision.set('Matching decision (at 0.01%FAR): ')
    lbl_decision.place(x=30, y=490)

    str_ptime = tkinter.StringVar()
    lbl_ptime = ttk.Label(rootWindow, textvariable=str_ptime, state='normal', font=("Helvetica", 12), borderwidth=0)
    str_ptime.set('Generation time: ')
    lbl_ptime.place(x=30, y=530)

    # cvlab logo
    tmp_image = Image.open('cvlab_logo.jpg')
    tmp_image = tmp_image.resize((190, 110), Image.BICUBIC)
    tmp_image = ImageTk.PhotoImage(tmp_image)
    place_for_result = ttk.Label(rootWindow, image=tmp_image, style='my.TButton', state='normal')
    place_for_result.image = tmp_image
    place_for_result.place(x=455, y=440)

    # d-fingerprint generation button
    btn_d_fp = ttk.Button(rootWindow, text='       d-fingerprint generation      ', command=d_fp_generation_run, style='my.TButton')
    btn_d_fp.place(x=20, y=380)

    # r-fingerprint generation button
    btn_d_fp = ttk.Button(rootWindow, text='       r-fingerprint generation       ', command=r_fp_generation_run, style='my.TButton')
    btn_d_fp.place(x=20, y=410)

    # liveness detection button
    # detect the liveness of the fingerprint according to
    # the specified sensor in different years.
    btn_r_fp = ttk.Button(rootWindow, text='        Liveness detection        ', command=livdet_run, style='my.TButton')
    btn_r_fp.place(x=250, y=380)
    # combobox
    bigfont = font.Font(family='Helvetica', size=13)
    rootWindow.option_add('*Font', bigfont)
    cb = ttk.Combobox(rootWindow, values=('2011_Biometrika',
                                          '2011_Digital',
                                          '2011_Italdata',
                                          '2011_Sagem',
                                          '2013_Biometrika',
                                          '2013_Italdata',
                                          '2015_CrossMatch',
                                          '2015_Digital_Persona',
                                          '2015_GreenBit',
                                          '2015_HiScan',
                                          '2019_Digital_Persona',
                                          '2019_GreenBit',
                                          'Unified',
                                          'User selection'))
    cb.set('2011_Biometrika')
    cb.place(x=250, y=412)
    cb.bind('<<ComboboxSelected>>', on_select)
    f = open('info.txt', 'w')
    f.write('2011_Biometrika.hdf5 0')
    f.close()


    # matching button
    btn_match = ttk.Button(rootWindow, text='               Matching               ', command=matching_run, style='my.TButton')
    btn_match.place(x=462, y=380)

    # exit button
    btn_exit = ttk.Button(rootWindow, text='                    Exit                   ', command=ExitApplication, style='my.TButton')
    btn_exit.place(x=462, y=410)

    rootWindow.mainloop()

if __name__ == '__main__':

    FP_matcher.ObtainLicenses()

    GUI()

