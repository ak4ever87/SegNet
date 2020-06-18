import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from Network import Network
from Params import Params
from utils import summary_tag_replace


def log_print(string):
    """
    Log a message to the standard output.
    :param string: The string to log.
    """
    now_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S: ')
    print(now_string + string)


tf_eps = tf.constant(1E-8)
HUBER_DELTA = 0.5


def dsc_loss(y_true, y_pred):
    """
    Dice Similarity Coefficient
    :param y_true: The ground-truth.
    :param y_pred: The predictions. (Softmax output)
    :return: The loss result.
    """
    return -2 * tf.reduce_sum((tf.multiply(y_true, y_pred))) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def smooth_l1(y_true, y_pred):
    """
    Smooth L1 Huber loss.
    :param y_true: The ground-truth.
    :param y_pred: The predictions. (Softmax output)
    :return: The loss result.
    """
    x = tf.abs(tf.to_float(y_true) - tf.to_float(y_pred))
    x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return tf.reduce_sum(x)


decay_bg = None
decay_fg = None
decay_edge = None


def train():
    """
    Train the model.
    """
    # Data input
    train_data_provider = params.train_data_provider
    val_data_provider = params.val_data_provider
    with tf.device('/cpu:0'):
        with tf.name_scope('Data'):
            train_image_batch, train_seg_batch, train_filenames = train_data_provider.get_batch(params.batch_size,
                                                                                                use_aug=True)
            val_image_batch, val_seg_batch, val_filenames = val_data_provider.get_batch(params.batch_size)
    # Build Network Graph
    net = Network(**params.net_build_params)
    val_net = Network(**params.net_build_params)

    def calc_losses(net_seg, gt_seg):
        """
        Calculate the losses to optimize.
        :param net_seg: The network to optimize.
        :param gt_seg: The batch to use.
        :return: The loss, Jaccard, pixel weighted loss, Local variables.
        """
        global decay_bg, decay_fg, decay_edge
        t_loss = []
        t_jaccard = []
        eps = tf.constant(np.finfo(np.float32).eps)
        with tf.name_scope('loss_calc'):
            gt_seg = tf.to_int32(tf.squeeze(gt_seg, axis=3))
            gt_valid = tf.to_float(tf.greater(gt_seg, -1))
            gt_bg = tf.to_float(tf.equal(gt_seg, 0))
            # Used for Jaccard
            gt_fg_bool = tf.equal(gt_seg, 1)
            gt_fg = tf.to_float(gt_fg_bool)
            gt_edge = tf.to_float(tf.equal(gt_seg, 2))
            pixel_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.maximum(gt_seg, 0),
                                                                        logits=net_seg)
            valid_pixel_loss = tf.multiply(pixel_loss, gt_valid)
            pixel_loss_weighted = 0
            loss_bg = tf.reduce_sum((valid_pixel_loss * gt_bg))
            loss_fg = tf.reduce_sum((valid_pixel_loss * gt_fg))
            loss_edge = tf.reduce_sum((valid_pixel_loss * gt_edge))
            total_loss = loss_bg + loss_fg + loss_edge + tf_eps
            if decay_bg is None:
                decay_bg = loss_bg / total_loss
            else:
                decay_bg = decay_bg * 0.99 + (loss_bg / total_loss) * 0.01
            if decay_fg is None:
                decay_fg = loss_fg / total_loss
            else:
                decay_fg = decay_fg * 0.99 + (decay_fg / total_loss) * 0.01
            if decay_edge is None:
                decay_edge = loss_edge / total_loss
            else:
                decay_edge = decay_edge * 0.99 + (decay_edge / total_loss) * 0.01
            cw = params.class_weights
            loss = (loss_bg * decay_bg * cw[0] + loss_fg * decay_fg * cw[1] + loss_edge * decay_edge * cw[2])
            # loss = (loss_bg * cw[0] + loss_fg * cw[1] + loss_edge * cw[2]) # Constant weight version.
            out_fg = tf.equal(tf.argmax(net_seg, 3), 1)
            intersection = tf.reduce_sum(tf.to_float(tf.logical_and(out_fg, gt_fg_bool)),
                                         axis=(1, 2), name='intersection')
            union = tf.reduce_sum(tf.to_float(tf.logical_or(out_fg, gt_fg_bool)), axis=(1, 2), name='union')

            jaccard = tf.reduce_mean(tf.divide(intersection + eps, union + eps, name='jaccard'))
            local_dict = {k: v for k, v in locals().items() if type(v) is tf.Tensor}
        return loss, jaccard, pixel_loss_weighted, local_dict

    device = '/gpu:0' if params.use_gpu else '/cpu:0'
    with tf.device(device):
        with tf.name_scope('train_tower'):
            with tf.variable_scope('net'):
                log_print('Building train network')
                norm_train_image_batch = tf.div(tf.subtract(train_image_batch, params.norm), params.norm)
                net_segs = net.build(norm_train_image_batch, training=True)
            log_print('Done building train network')
            net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='net')
            train_loss, train_jaccard, _, _ = calc_losses(net_segs, train_seg_batch)
            # opt = tf.train.RMSPropOptimizer(params.learning_rate)
            opt = tf.train.AdamOptimizer(params.learning_rate)
            grads_and_vars = opt.compute_gradients(train_loss, net_vars)
            global_step = tf.Variable(0, trainable=False)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_step = opt.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        with tf.name_scope('val_tower'):
            with tf.variable_scope('net', reuse=True):
                log_print('Building validation network')
                norm_val_image_batch = tf.div(tf.subtract(val_image_batch, params.norm), params.norm)
                valnet_segs = val_net.build(norm_val_image_batch, training=False)
                log_print('Done building validation network')
                val_loss, val_jaccard, _, _ = calc_losses(valnet_segs, val_seg_batch)

    # Tensorboard

    # Train Summaries

    tf.summary.image('Image', train_image_batch, max_outputs=1, collections=['train_summaries'])

    train_out_img = tf.nn.softmax(net_segs)
    tf.summary.image('Segmentation', train_out_img, max_outputs=1, collections=['train_summaries'])
    tf.summary.image('Ground Truth', train_seg_batch, max_outputs=1, collections=['train_summaries'])
    tf.summary.scalar('Loss', train_loss, collections=['train_summaries'])
    tf.summary.scalar('Jaccard', train_jaccard, collections=['train_summaries'])
    tf.summary.scalar('Learning Rate', params.learning_rate, collections=['train_summaries'])

    # Val Summaries
    with tf.name_scope('tb_val'):
        tf.summary.image('Image', val_image_batch, max_outputs=1, collections=['val_summaries'])
        val_out_img = tf.nn.softmax(valnet_segs)
        tf.summary.image('Segmentation', val_out_img, max_outputs=1, collections=['val_summaries'])

        tf.summary.image('Ground Truth', val_seg_batch, max_outputs=1, collections=['val_summaries'])
        tf.summary.scalar('Loss', val_loss, collections=['val_summaries'])
        tf.summary.scalar('Jaccard', val_jaccard, collections=['val_summaries'])

    q_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='Data')
    custom_train_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train_tower')
    custom_val_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='val_tower')
    train_summaries = tf.summary.merge(tf.get_collection('train_summaries') + custom_train_summaries + q_summaries)
    val_summaries = tf.summary.merge(tf.get_collection('val_summaries') + custom_val_summaries)

    summaries_dir = params.experiment_log_dir
    train_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'train'), graph=tf.get_default_graph())
    val_writer = tf.summary.FileWriter(os.path.join(summaries_dir, 'val'))

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=params.save_checkpoint_max_to_keep,
                           keep_checkpoint_every_n_hours=params.save_checkpoint_every_N_hours)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    coord = tf.train.Coordinator()
    with tf.Session(config=config) as sess:
        log_print('Started TF Session')
        sess.run(init_op)
        log_print('Variables Initialize')
        if params.load_checkpoint:
            saver.restore(sess, params.load_checkpoint_path)
            log_print('Params loaded from checkpoint: {}'.format(params.load_checkpoint_path))

        t = sess.run(global_step)
        threads = tf.train.start_queue_runners(sess, coord)
        log_print('Started {} data threads'.format(len(threads)))
        elapsed_time = 0.
        end_time = 0.
        other_time = 0.
        if params.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            try:
                os.makedirs(os.path.join(summaries_dir, 'profile'))
            except OSError:
                pass
        else:
            options = tf.RunOptions()

        run_metadata = tf.RunMetadata()
        feed_dict = {}
        val_feed_dict = {}
        while t < params.num_iterations:
            try:
                start_time = time.time()
                other_time += start_time - end_time
                _, t, train_loss_eval, train_jaccard_eval, train_summaries_eval = sess.run(
                    [train_step, global_step, train_loss, train_jaccard, train_summaries],
                    feed_dict=feed_dict,
                    options=options,
                    run_metadata=run_metadata)
                end_time = time.time()
                elapsed_time += end_time - start_time
                if params.profile:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(os.path.join(summaries_dir, 'profile',
                                           'timeline_d{}.json'.format(t)), 'w') as f:
                        f.write(chrome_trace)

                if not t % 10:
                    s = 'Iteration {0}: [Train] Loss: {1:.3f}, Jaccard: {2:.3f}, Time: {3:.1f} seconds. '
                    log_print(s.format(t,
                                       train_loss_eval,
                                       train_jaccard_eval,
                                       elapsed_time / 10.0))
                    elapsed_time = 0.
                    other_time = 0.

                if not t % params.validation_interval:
                    vout, val_loss_eval, val_jaccard_eval, val_summaries_eval = sess.run(
                        [val_out_img, val_loss, val_jaccard,
                         val_summaries],
                        feed_dict=val_feed_dict)

                    s = 'Iteration {0}: [Val] Loss: {1:.3f}, Jaccard: {2:.3f}, Time: {3:.1f} seconds. '
                    log_print(s.format(t,
                                       val_loss_eval,
                                       val_jaccard_eval,
                                       elapsed_time / 10.0))

                    if not params.dry_run:
                        val_summaries_eval = summary_tag_replace(val_summaries_eval, 'tb_val/', '')
                        val_writer.add_summary(val_summaries_eval, t)

                if not params.dry_run:
                    if not t % params.save_checkpoint_iteration:
                        save_path = saver.save(sess, os.path.join(params.experiment_save_dir,
                                                                  "model_%d.ckpt") % t)
                        log_print('Model saved to path: {}'.format(save_path))
                    if not t % params.write_to_tb_interval:
                        train_writer.add_summary(train_summaries_eval, t)
                        log_print('Printed iteration {} to Tensorboard'.format(t))
            except (ValueError, RuntimeError, KeyboardInterrupt):
                if not params.dry_run:
                    save_path = saver.save(sess, os.path.join(params.experiment_save_dir,
                                                              "model_%d.ckpt") % t)
                    log_print('Model saved to path: {}'.format(save_path))

                coord.request_stop()
                coord.join(threads)
                return
        coord.request_stop()
        coord.join(threads)
        if not params.dry_run:
            save_path = saver.save(sess, os.path.join(params.experiment_save_dir,
                                                      "model_final.ckpt") % t)
            log_print('Model saved to path: {}'.format(save_path))


if __name__ == '__main__':
    params = Params()
    train()
