#coding=utf-8
import tensorflow as tf

IMAGE_SIZE = 32

NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 18079
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5000


def _read_anim(filename, batch_size):

    class AnimRecord(object):
        pass

    # 这个是一个一个的读取训练数据
    # for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    #     example = tf.train.Example()
    #     example.ParseFromString(serialized_example)
    #     image = example.features.feature['image'].bytes_list.value
    #     label = example.features.feature['label'].int64_list.value

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string)
    })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return image, label


def eval_data(file_name, batch_size):
    images, labels = _read_anim(file_name, batch_size)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # crop to 24 * 24
    # distorted_image = tf.random_crop(images, [width, height, 3])
    float_image = tf.image.per_image_standardization(images)
    return _generate_image_and_label_batch(float_image, labels, 20000, batch_size)


def input_data(file_name, batch_size):
    images, labels = _read_anim(file_name, batch_size)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # crop to 24 * 24
    # distorted_image = tf.random_crop(images, [width, height, 3])
    distorted_image = tf.image.random_flip_left_right(images)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    return _generate_image_and_label_batch(float_image, labels, 20000, batch_size)


def _generate_image_and_label_batch(image, label, min_queue_example, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_example + 3 * batch_size,
        min_after_dequeue=min_queue_example
    )
    return images, tf.reshape(label_batch, [batch_size])