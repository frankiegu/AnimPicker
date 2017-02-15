# encoding=utf-8
import tensorflow as tf

a = {}

b = []

d = 0
fd = 0

for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    image = example.features.feature['image'].bytes_list.value
    label = example.features.feature['label'].int64_list.value

    # 可以做一些预处理之类的
    k = len(image[0])
    if k in a.keys():
        a[k] += 1
    else:
        a[k] = 1

    b.append(label[0])
    if label[0] == 0:
        d += 1
    else:
        fd += 1

print d, fd

# filename_queue = tf.train.string_input_producer(["train.tfrecords"])
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
# features = tf.parse_single_example(serialized_example, features={
#     'label': tf.FixedLenFeature([], tf.int64),
#     'image': tf.FixedLenFeature([], tf.string)
# })
#
# image = tf.decode_raw(features['image'], tf.uint8)
# image = tf.reshape(image, [32, 32, 3])
# image = tf.cast(image, tf.float32)
# label = tf.cast(features['label'], tf.int32)
#
# IMAGE_SIZE = 24
#
# height = IMAGE_SIZE
# width = IMAGE_SIZE
#
# # crop to 24 * 24
# distorted_image = tf.random_crop(image, [width, height, 3])
# distorted_image = tf.image.random_flip_left_right(distorted_image)
# distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
# distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
# float_image = tf.image.per_image_standardization(distorted_image)
#
# with tf.Session() as sess:
#     print sess.run(distorted_image)