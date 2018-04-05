import tensorflow as tf


def causal_conv(value, filter_width, filter_num, dilation, name='causal_conv'):
    with tf.name_scope(name):
        value = tf.pad(value, [[0, 0], [dilation*(filter_width-1), 0],
                               [0, 0]], mode='CONSTANT', constant_values=0.0)
        result = tf.layers.conv1d(inputs=value, filters=filter_num,
                                  kernel_size=filter_width, dilation_rate=dilation,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  padding='valid')
        return result


def bi_causal_conv(value, filter_width, filter_num, dilation, name='bi_causal_conv'):
    with tf.name_scope(name):
        fw = causal_conv(value, filter_width, filter_num, dilation, name='fw')
        bw = causal_conv(tf.reverse(value, [1]), filter_width, filter_num, dilation, name='bw')
        return tf.concat([fw, tf.reverse(bw, [1])], axis=-1)


# if __name__ == '__main__':
#     import numpy as np
#     a = 1.0*np.random.randint(0, 10, [2, 7, 2])
#     a[1][-3:][:] = 0.0
#     seq_length = np.array([7, 4])
#     print('a', a)
#     with tf.Session() as sess:
#         a = tf.constant(a, tf.float32)
#         # a = tf.pad(a, [[0, 0], [2, 0], [0, 0]])
#         b = causal_conv(a, 3, 2, 1)
#         c = causal_conv(a, 3, 2, 2)
#         d = bi_causal_conv(a, 3, 2, 2, seq_length)
#         sess.run(tf.global_variables_initializer())
#         # print(sess.run(b))
#         # # print(sess.run(batch_to_time(time_to_batch(a, 2), 2)))
#         # print(sess.run(c))
#         print(tf.reverse_sequence(a, seq_length, 1, 0).eval())
#         print(sess.run(d))
