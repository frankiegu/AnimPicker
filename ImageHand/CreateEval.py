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

    writer = tf.python_io.TFRecordWriter("eval.tfrecords")
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
        '风景_小动物_thumb',
        '番剧动漫_thumb'
    ]
    labels = [
        1,
        0
    ]
    # print len(os.listdir(paths[0]))
    # print len(os.listdir(paths[1]))
    main(paths, labels)