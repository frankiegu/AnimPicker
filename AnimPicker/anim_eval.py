# coding=utf-8
from datetime import datetime
import os
import numpy as np
import math
import tensorflow as tf
import time

import anim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("eval_dir", "anim_eval", "eval dir")
tf.app.flags.DEFINE_string('eval_data', 'eval.tfrecords', "用于eval的数据")
tf.app.flags.DEFINE_string('check_point_dir', 'anim_trian/', "check point dir")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 3, "how often to run eval")
tf.app.flags.DEFINE_integer('num_examples', 3720, "数量")
tf.app.flags.DEFINE_boolean('eval_once', False, "运行一次")


def eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.check_point_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print "no checkpoint found"

        print "start"

        # start the queue runner
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess,
                                                 coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                prefictions, labs, lo = sess.run([top_k_op, labels, logits])

                # print lo
                # print labs

                true_count += np.sum(prefictions)
                step += 1

            precision = true_count * 1.0 / total_sample_count
            print "total_sample_count:" + str(total_sample_count)
            print "true_count:" + str(true_count)
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            print e
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def main():
    with tf.Graph().as_default():
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data
        images, labels = anim.eval_data(FLAGS.eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = anim.interface(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            anim.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                               graph_def=graph_def)

        print "start eval"
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op, logits, labels)
            if FLAGS.eval_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

if __name__ == '__main__':
    main()