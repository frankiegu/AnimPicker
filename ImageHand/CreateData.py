# encoding=utf-8
import os
import io
import sys
import tensorflow as tf
from PIL import Image
import imghdr


def main(paths, labels):
    if not paths:
        print "请输入图片文件夹"
        return

    if not labels:
        print "请输入标签"
        return

    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for i in range(len(paths)):
        path = paths[i]
        label = labels[i]
        print path
        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            if not imghdr.what(image_path):
                continue

            img = Image.open(image_path)
            # byte_io = io.BytesIO()
            # img.save(byte_io, format="JPEG")
            # img_raw = byte_io.getvalue()
            try:
                img_raw = img.tobytes()
            except AttributeError as e:
                img_raw = img.tostring()
            if not len(img_raw) == 3072:
                print "长度不是3072,是" + str(len(img_raw))
                continue
            example = tf.train.Example(features=tf.train.Features(feature={
                             "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                             }))
            writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    paths = [
        '二次元动漫_thumb',
        '城市_thumb',
        '动漫_thumb',
        '宠物_thumb',
        '动漫人物_thumb',
        '狗_thumb',
        '日本动漫_thumb',
        '真人_thumb',
        '高清动漫_thumb',
        '风景_thumb'
    ]
    labels = [
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1
    ]
    main(paths, labels)