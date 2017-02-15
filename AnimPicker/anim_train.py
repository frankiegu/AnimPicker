#coding=utf-8
import tensorflow as tf
import anim
import anim_input

import numpy as np
import time
from datetime import datetime
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'anim_trian/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        images, labels = anim.distored_input()

        logits = anim.interface(images)

        loss = anim.loss(logits, labels)

        train_op = anim.train(loss, global_step)

        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement
        ))

        init = tf.global_variables_initializer()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "restore from file"
        else:
            print "no checkpoint found"

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), "Model diverged with loss = NaN"

            if step % 10 == 0:
                num_example_per_step = FLAGS.batch_size
                examples_per_sec = num_example_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value,
                                     examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step % 500 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == "__main__":
    train()