import datetime
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs  # pylint: disable=no-name-in-module
from tensorflow.python.ops import array_ops, rnn  # pylint: disable=no-name-in-module

from data_utils.get_chunks import get_chunks
from model.base_model import BaseModel
from model.configs import from_json, Config
from model.causal_conv import bi_causal_conv


def stack_bidirectional_dynamic_rnn_cnn(cells_fw,
                                        cells_bw,
                                        cnn_filters,
                                        cnn_sizes,
                                        dropout,
                                        inputs,
                                        initial_states_fw=None,
                                        initial_states_bw=None,
                                        dtype=None,
                                        sequence_length=None,
                                        parallel_iterations=None,
                                        time_major=False,
                                        scope=None):
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw, filters, size) in enumerate(zip(cells_fw, cells_bw, cnn_filters, cnn_sizes)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]

            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    time_major=time_major)
                # Concat the outputs to create the new input.
                prev_layer = array_ops.concat(outputs, 2)
            with vs.variable_scope("conv_%d" % i):
                prev_layer = tf.layers.conv1d(
                    inputs=prev_layer,
                    filters=filters,
                    kernel_size=size,
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                )
                if dropout is not None:
                    prev_layer = tf.nn.dropout(prev_layer, dropout)
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def stack_bidirectional_dynamic_rnn(cells_fw,
                                    cells_bw,
                                    inputs,
                                    concat_residual=False,
                                    initial_states_fw=None,
                                    initial_states_bw=None,
                                    dtype=None,
                                    sequence_length=None,
                                    parallel_iterations=None,
                                    time_major=False,
                                    scope=None):
    states_fw = []
    states_bw = []
    prev_layer = inputs

    with vs.variable_scope(scope or "stack_bidirectional_rnn"):
        for i, (cell_fw, cell_bw) in enumerate(zip(cells_fw, cells_bw)):
            initial_state_fw = None
            initial_state_bw = None
            if initial_states_fw:
                initial_state_fw = initial_states_fw[i]
            if initial_states_bw:
                initial_state_bw = initial_states_bw[i]
            with vs.variable_scope("cell_%d" % i):
                outputs, (state_fw, state_bw) = rnn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    prev_layer,
                    initial_state_fw=initial_state_fw,
                    initial_state_bw=initial_state_bw,
                    sequence_length=sequence_length,
                    parallel_iterations=parallel_iterations,
                    dtype=dtype,
                    time_major=time_major)
                if concat_residual:
                    prev_layer = array_ops.concat(outputs+(prev_layer,), 2)
                else:
                    prev_layer = tf.layers.conv1d(inputs=array_ops.concat(outputs, 2),
                                                  filters=prev_layer.get_shape(
                    )[-1],
                        kernel_size=1,
                        activation=None,
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    ) + prev_layer
            states_fw.append(state_fw)
            states_bw.append(state_bw)

    return prev_layer, tuple(states_fw), tuple(states_bw)


def build_gru_cell(hidden):
    return tf.nn.rnn_cell.GRUCell(
        hidden,
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        bias_initializer=tf.constant_initializer(0.0, tf.float32)
    )


def build_lstm_layer_norm(hidden, keep_prob):
    return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=hidden,
                                                 forget_bias=1.0,
                                                 activation=tf.tanh,
                                                 layer_norm=True,
                                                 norm_gain=1.0,
                                                 norm_shift=0.0,
                                                 dropout_keep_prob=keep_prob,
                                                 dropout_prob_seed=None,
                                                 reuse=None
                                                 )


def build_gru_cell_with_dropout(hidden, keep_prob):
    return tf.nn.rnn_cell.DropoutWrapper(
        build_gru_cell(hidden),
        output_keep_prob=keep_prob,
        variational_recurrent=True,
        dtype=tf.float32
    )


def load_model(path):
    model = SequenceTaggingModel(Config())
    model.sess = tf.Session()
    saver = tf.train.import_meta_graph("{}.meta".format(path))
    saver.restore(model.sess, path)
    graph = tf.get_default_graph()
    model.word_ids = graph.get_operation_by_name('word_ids').outputs[0]
    model.char_ids = graph.get_operation_by_name('char_ids').outputs[0]
    model.sequence_length = graph.get_operation_by_name(
        'sequence_length').outputs[0]
    model.word_length = graph.get_operation_by_name('word_length').outputs[0]
    model.dropout = graph.get_operation_by_name('dropout').outputs[0]
    try:
        model.transition_params = graph.get_operation_by_name(
            'transitions').outputs[0]
        model.decode_tags = graph.get_operation_by_name(
            'decode_tags').outputs[0]
        model.best_scores = graph.get_operation_by_name(
            'best_scores').outputs[0]
        print(model.sess.run(graph.get_operation_by_name(
            'word_embedding/mul_3/x').outputs[0]))
    except KeyError:
        model.label_preds = graph.get_operation_by_name(
            'label_preds').outputs[0]
    return model


def build_with_params(path, mode='train'):
    # with open(path, 'r') as file:
    #     hp = json.load(file)
    # configs = MockConfigs()
    # configs.__dict__.update(**hp)
    config = from_json(path)
    model = SequenceTaggingModel(config, mode=mode)
    if not mode == 'train':
        mock_embedding = np.zeros([config.nwords, config.wdims], dtype=float)
        model.build_model(mock_embedding)
    return model


def load_and_save_model(hp_path, weight_path, out_path, version):
    model = build_with_params(hp_path, 'infer')
    model.sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(model.sess, weight_path)
    export_path = os.path.join(
        tf.compat.as_bytes(out_path),
        tf.compat.as_bytes(str(version))
    )
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    word_ids_info = tf.saved_model.utils.build_tensor_info(model.word_ids)
    char_ids_info = tf.saved_model.utils.build_tensor_info(model.char_ids)
    sequence_length_info = tf.saved_model.utils.build_tensor_info(
        model.sequence_length)
    word_length_info = tf.saved_model.utils.build_tensor_info(
        model.word_length)
    decode_tags_info = tf.saved_model.utils.build_tensor_info(
        model.decode_tags)
    best_scores_info = tf.saved_model.utils.build_tensor_info(
        model.best_scores)
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            'word_ids': word_ids_info,
            'char_ids': char_ids_info,
            'sequence_length': sequence_length_info,
            'word_length': word_length_info
        },
        outputs={
            'decode_tags': decode_tags_info,
            'best_scores': best_scores_info
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    builder.add_meta_graph_and_variables(
        model.sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "sequence_tags": prediction_signature
        }
    )
    builder.save(as_text=False)


class SequenceTaggingModel(BaseModel):
    def __init__(self, configs, mode='train'):
        super().__init__(configs)
        self.mode = mode

    def _add_placeholders(self):
        # Word input sequence [batch_size x max_sequence_length]
        self.word_ids = tf.placeholder(tf.int32, [None, None], name='word_ids')

        # Character input sequence [batch_size x max_sequence_length x max_word_length]
        self.char_ids = tf.placeholder(
            tf.int32, [None, None, None],
            name='char_ids'
        )

        # Word input sequence length [batch_size]
        self.sequence_length = tf.placeholder(
            tf.int32, [None],
            name='sequence_length'
        )

        # Word lengths [batch_size x max_sequence_length]
        self.word_length = tf.placeholder(
            tf.int32, [None, None],
            name='word_length'
        )

        # Sequence labels for training [batch_size x max_sequence_length]
        self.labels = tf.placeholder(
            tf.int32, [None, None],
            name='labels'
        )
        self.loss = 0
        if self.mode == 'train':
            self.dropout = tf.placeholder(tf.float32, [], name='dropout')
            self.rnn_cell = (lambda hidden: build_lstm_layer_norm(hidden, self.dropout)) if self.configs.lstm_layer_norm else (lambda hidden: build_gru_cell_with_dropout(
                hidden, self.dropout))
        else:
            self.dropout = None
            self.rnn_cell = build_lstm_layer_norm if self.configs.lstm_layer_norm else build_gru_cell

    def build_model(self, pretrain_word_embedding):
        self._add_placeholders()
        self._build_word_embedding(pretrain_word_embedding)
        self._add_logits_op()
        self._add_loss_op()
        self._add_pred_op()
        if self.mode == 'train':
            self._add_train_op(method=self.configs.training_method, loss=self.loss,
                               learning_rate=self.configs.learning_rate,
                               clip=self.configs.clip_grad)
            self._initialize_session()

    def save_hyperparams(self):
        hp = {
            'nwords': self.configs.nwords,
            'wdims': self.configs.wdims,
            'num_classes': self.configs.num_classes,
            'num_hidden_word': self.configs.num_hidden_word,
            'num_hidden_char': self.configs.num_hidden_char,
            'nchars': self.configs.nchars,
            'cdims': self.configs.cdims,
            'train_word_embedding': self.configs.train_word_embedding,
            'use_crf': self.configs.use_crf,
            'sum_vector': self.configs.sum_vector,
            'word_char_cosine': self.configs.word_char_cosine,
            'use_conv': self.configs.use_conv,
            'num_filters': self.configs.num_filters,
            'kernel_size': self.configs.kernel_size,
            'char_embedding': self.configs.char_embedding,
            'char_embedding_kernel_size': self.configs.char_embedding_kernel_size,
            'use_rnn': self.configs.use_rnn,
            'use_residual': self.configs.use_residual,
            'clip_grad': self.configs.clip_grad,
            'dilation_rate': self.configs.dilation_rate,
            'final_layer': self.configs.final_layer,
            'final_layer_kernel': self.configs.final_layer_kernel,
            'latent_dim': self.configs.latent_dim,
            'final_layer_filters': self.configs.final_layer_filters,
            'lstm_layer_norm': self.configs.lstm_layer_norm,
            'stack_gru': self.configs.stack_gru,
            'stack_rnn_cnn': self.configs.stack_rnn_cnn,
            'concat_residual': self.configs.concat_residual,
            'use_bi_causal_conv': self.configs.use_bi_causal_conv,
            'bi_causal_conv_block': self.configs.bi_causal_conv_block,
            'block_rnn': self.configs.block_rnn,
            'num_block_rnn': self.configs.num_block_rnn
        }
        with open(os.path.join(self.configs.out_dir, 'hyperparams.json'), 'w') as file:
            json.dump(hp, file)

    def _add_pred_op(self):
        if self.configs.use_crf:
            self.decode_tags, self.best_scores = tf.contrib.crf.crf_decode(
                potentials=self.logits,
                transition_params=self.transition_params,
                sequence_length=self.sequence_length
            )
            self.decode_tags = tf.identity(self.decode_tags, 'decode_tags')
            self.best_scores = tf.identity(self.best_scores, 'best_scores')
        else:
            self.label_preds = tf.argmax(
                self.logits, axis=-1, output_type=tf.int32, name='label_preds')

    def _build_word_embedding(self, pretrain_word_embedding):
        with tf.variable_scope('word_embedding'):
            if pretrain_word_embedding is None:
                word_embedding = tf.Variable(
                    tf.random_uniform(
                        [self.configs.nwords, self.configs.wdims],
                        -0.1, 0.1,
                        tf.float32
                    )
                )
            else:
                self.configs.nwords, self.configs.wdims = pretrain_word_embedding.shape
                word_embedding = tf.Variable(
                    pretrain_word_embedding,
                    dtype=tf.float32,
                    trainable=self.configs.train_word_embedding
                )
            word_embedding = tf.nn.embedding_lookup(
                word_embedding,
                self.word_ids,
                name='word_embedding'
            )
            char_embedding = tf.Variable(
                tf.random_uniform(
                    [self.configs.nchars, self.configs.cdims],
                    -0.1, 0.1,
                    tf.float32
                )
            )
            char_embedding = tf.nn.embedding_lookup(
                char_embedding, self.char_ids,
                name='char_embedding'
            )
            s = tf.shape(char_embedding)
            char_embedding = tf.reshape(
                char_embedding,
                [s[0]*s[1], s[2], self.configs.cdims]
            )
            if self.configs.char_embedding == 'rnn':
                word_length = tf.reshape(self.word_length, [s[0]*s[1]])
                _, fs_fw, fs_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    [self.rnn_cell(x) for x in self.configs.num_hidden_char],
                    [self.rnn_cell(x) for x in self.configs.num_hidden_char],
                    char_embedding,
                    sequence_length=word_length,
                    dtype=tf.float32
                )
                output = tf.concat([fs_fw[-1], fs_bw[-1]], axis=-1)
                output = tf.tanh(output)
            else:
                output = tf.layers.conv1d(
                    inputs=char_embedding,
                    filters=2*self.configs.num_hidden_char[-1],
                    kernel_size=self.configs.char_embedding_kernel_size,
                    padding='same',
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                output = tf.reduce_max(output, axis=-2)
            if self.configs.sum_vector:
                W0 = tf.get_variable(
                    'W0', shape=[2*self.configs.num_hidden_char[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                output = tf.tanh(tf.matmul(output, W0))
                word_embedding = tf.reshape(
                    word_embedding, [s[0]*s[1], pretrain_word_embedding.shape[-1]])
                g = np.ones(pretrain_word_embedding.shape[0])
                g[-1] = 0
                g = tf.nn.embedding_lookup(
                    tf.constant(g, dtype=tf.float32),
                    self.word_ids,
                    name='g'
                )
                self.loss += self.configs.word_char_cosine * tf.reduce_sum(g*tf.losses.cosine_distance(
                    tf.stop_gradient(tf.nn.l2_normalize(
                        word_embedding, -1)),
                    tf.nn.l2_normalize(output, -1),
                    -1
                ))
                W1 = tf.get_variable(
                    'W1', shape=[pretrain_word_embedding.shape[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                W2 = tf.get_variable(
                    'W2', shape=[pretrain_word_embedding.shape[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                W3 = tf.get_variable(
                    'W3', shape=[pretrain_word_embedding.shape[-1], pretrain_word_embedding.shape[-1]],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
                z = tf.sigmoid(tf.matmul(tf.tanh(tf.add(tf.matmul(word_embedding, W1),
                                                        tf.matmul(output, W2))), W3))
                word_embedding = tf.add(tf.multiply(z, word_embedding),
                                        tf.multiply((1-z), output))
                word_embedding = tf.reshape(
                    word_embedding, [s[0], s[1],
                                     pretrain_word_embedding.shape[-1]]
                )
            else:
                output = tf.reshape(
                    output, [s[0], s[1], 2*self.configs.num_hidden_char[-1]]
                )
                word_embedding = tf.concat([word_embedding, output], axis=-1)
            self.word_embedding = tf.nn.dropout(
                word_embedding, self.dropout) if self.mode == 'train' else word_embedding

    def _add_logits_op(self):
        if self.configs.use_rnn:
            if self.configs.use_conv and not self.configs.stack_rnn_cnn:
                with tf.variable_scope('mask_cnn'):
                    mask = tf.tile(tf.expand_dims(tf.cast(tf.not_equal(self.word_ids, 0),
                                                          tf.float32),
                                                  axis=-1),
                                   multiples=[
                        1, 1, self.word_embedding.get_shape()[-1]]
                    )
                    inputs = self.word_embedding * mask
                if type(self.configs.kernel_size) is list:
                    rnn_inputs = []
                    for kernel_size, filters in zip(self.configs.kernel_size, self.configs.num_filters):
                        rnn_inputs.append(tf.expand_dims(tf.layers.conv1d(inputs,
                                                                          filters=filters,
                                                                          kernel_size=kernel_size,
                                                                          strides=1,
                                                                          padding='same',
                                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                                          activation=tf.nn.relu), axis=-1))
                    rnn_input = tf.concat(rnn_inputs, axis=-1)
                    rnn_input = tf.layers.conv2d(
                        inputs=rnn_input,
                        filters=1,
                        kernel_size=(1, 1),
                        activation=tf.nn.relu
                    )
                    rnn_input = tf.squeeze(rnn_input, axis=-1)
                else:
                    rnn_input = tf.layers.conv1d(inputs,
                                                 filters=self.configs.num_filters,
                                                 kernel_size=self.configs.kernel_size,
                                                 strides=1,
                                                 padding='same',
                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                 activation=tf.nn.relu)
                # if self.mode == 'train':
                #     rnn_input = tf.nn.dropout(
                #         rnn_input, keep_prob=self.dropout)
            else:
                rnn_input = self.word_embedding
            if self.configs.bi_gru:
                if self.configs.stack_gru:
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.nn.rnn_cell.MultiRNNCell(
                            cells=[self.rnn_cell(x) for x in self.configs.num_hidden_word]),
                        cell_bw=tf.nn.rnn_cell.MultiRNNCell(
                            cells=[self.rnn_cell(x) for x in self.configs.num_hidden_word]),
                        inputs=rnn_input,
                        sequence_length=self.sequence_length,
                        dtype=tf.float32
                    )
                else:
                    if self.configs.stack_rnn_cnn:
                        if self.configs.use_conv:
                            rnn_input = tf.layers.conv1d(self.word_embedding,
                                                         filters=self.configs.num_filters[0],
                                                         kernel_size=self.configs.kernel_size[0],
                                                         strides=1,
                                                         padding='same',
                                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                         activation=tf.nn.relu)
                            filters = self.configs.num_filters[1:]
                            kernel_sizes = self.configs.kernel_size[1:]
                        else:
                            filters = self.configs.num_filters
                            kernel_sizes = self.configs.kernel_size
                        outputs, _, _ = stack_bidirectional_dynamic_rnn_cnn(
                            cells_fw=[self.rnn_cell(x)
                                      for x in self.configs.num_hidden_word],
                            cells_bw=[self.rnn_cell(x)
                                      for x in self.configs.num_hidden_word],
                            cnn_filters=filters,
                            cnn_sizes=kernel_sizes,
                            dropout=self.dropout,
                            inputs=rnn_input,
                            dtype=tf.float32
                        )
                        final_size = self.configs.num_filters[-1]
                    else:
                        if self.configs.block_rnn:
                            dynamic_rnn = tf.contrib.rnn.stack_bidirectional_dynamic_rnn
                            outputs = rnn_input
                            for i in range(self.configs.num_block_rnn):
                                temp = outputs
                                with tf.variable_scope('block_rnn_%d' % i):
                                    outputs, _, _ = dynamic_rnn(
                                        [self.rnn_cell(x)
                                         for x in self.configs.num_hidden_word],
                                        [self.rnn_cell(x)
                                         for x in self.configs.num_hidden_word],
                                        rnn_input,
                                        sequence_length=self.sequence_length,
                                        dtype=tf.float32
                                    )
                                    if self.configs.use_residual:
                                        if self.configs.concat_residual:
                                            outputs = array_ops.concat(
                                                [outputs, temp], 2)
                                        else:
                                            outputs = tf.layers.conv1d(inputs=array_ops.concat(outputs, 2),
                                                                       filters=temp.get_shape(
                                            )[-1],
                                                kernel_size=1,
                                                activation=None,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            ) + temp
                                    final_size = outputs.get_shape()[-1]
                        else:
                            dynamic_rnn = lambda *args, **kwargs: stack_bidirectional_dynamic_rnn(
                                concat_residual=self.configs.concat_residual, *args, **kwargs) if self.configs.use_residual else tf.contrib.rnn.stack_bidirectional_dynamic_rnn
                            outputs, _, _ = dynamic_rnn(
                                [self.rnn_cell(x)
                                 for x in self.configs.num_hidden_word],
                                [self.rnn_cell(x)
                                 for x in self.configs.num_hidden_word],
                                rnn_input,
                                sequence_length=self.sequence_length,
                                dtype=tf.float32
                            )
                            final_size = 2*self.configs.num_hidden_word[-1]
            else:
                outputs, _ = tf.nn.dynamic_rnn(
                    cell=tf.nn.rnn_cell.MultiRNNCell(
                        cells=[self.rnn_cell(x) for x in self.configs.num_hidden_word]),
                    inputs=rnn_input,
                    sequence_length=self.sequence_length,
                    dtype=tf.float32
                )
                final_size = self.configs.num_hidden_word[-1]
        else:
            with tf.variable_scope('mask_cnn'):
                mask = tf.tile(tf.expand_dims(tf.cast(tf.not_equal(self.word_ids, 0),
                                                      tf.float32),
                                              axis=-1),
                               multiples=[
                    1, 1, self.word_embedding.get_shape()[-1]]
                )
                inputs = self.word_embedding * mask
            if self.configs.use_bi_causal_conv:
                outputs = inputs
                i = 0
                if self.configs.bi_causal_conv_block:
                    for j in range(self.configs.bi_causal_conv_block):
                        with tf.variable_scope('bi_causal_conv_block_%d' % j):
                            temp = outputs
                            i = 0
                            for filters, dilation_rate, kernel_size in zip(self.configs.num_hidden_word, self.configs.kernel_size, self.configs.dilation_rate):
                                outputs = bi_causal_conv(
                                    outputs, kernel_size, filters, dilation_rate, name='bi_causal_conv_%d' % i)
                                i += 1
                            if self.configs.use_residual:
                                outputs = tf.layers.conv1d(
                                    outputs,
                                    filters=temp.get_shape()[-1],
                                    kernel_size=1,
                                    strides=1,
                                    padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    name='conv1d_res_{}'.format(i),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                                ) + temp
                            final_size = outputs.get_shape()[-1]
                            if self.mode == 'train':
                                outputs = tf.nn.dropout(outputs, self.dropout)
                else:
                    for filters, dilation_rate, kernel_size in zip(self.configs.num_hidden_word, self.configs.kernel_size, self.configs.dilation_rate):
                        temp = outputs
                        outputs = bi_causal_conv(
                            outputs, kernel_size, filters, dilation_rate, name='bi_causal_conv_%d' % i)
                        final_size = outputs.get_shape()[-1]
                        if self.configs.use_residual:
                            outputs = tf.layers.conv1d(
                                outputs,
                                filters=temp.get_shape()[-1],
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                activation=tf.nn.relu,
                                name='conv1d_res_{}'.format(i),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                    self.configs.l2_regs)
                            ) + temp
                        i += 1
                        if self.mode == 'train':
                            outputs = tf.nn.dropout(outputs, self.dropout)
            else:
                outputs = tf.layers.conv1d(
                    inputs=inputs,
                    filters=self.configs.latent_dim,
                    kernel_size=1,
                    activation=tf.nn.relu,
                    strides=1
                )
                if self.configs.bi_causal_conv_block:
                    for j in range(self.configs.bi_causal_conv_block):
                        with tf.variable_scope('bi_causal_conv_block_%d' % j):
                            temp = outputs
                            for i, num_filters in enumerate(self.configs.num_hidden_word):
                                outputs = tf.layers.conv1d(
                                    outputs,
                                    filters=num_filters,
                                    kernel_size=self.configs.kernel_size[i],
                                    strides=1,
                                    padding='same',
                                    dilation_rate=self.configs.dilation_rate[i],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    name='conv1d_word_{}'.format(i),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                                )
                            if self.configs.use_residual:
                                if self.configs.concat_residual:
                                    outputs = array_ops.concat(
                                        [outputs, temp], axis=-1)
                                else:
                                    outputs = tf.layers.conv1d(
                                        outputs,
                                        filters=temp.get_shape()[-1],
                                        kernel_size=1,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        activation=None,
                                        name='conv1d_res_{}'.format(i),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                            self.configs.l2_regs)
                                    ) + temp
                            final_size = outputs.get_shape()[-1]
                            if self.mode == 'train':
                                outputs = tf.nn.dropout(outputs, self.dropout)
                else:

                    for i, num_filters in enumerate(self.configs.num_hidden_word):
                        temp = outputs
                        outputs = tf.layers.conv1d(
                            outputs,
                            filters=num_filters,
                            kernel_size=self.configs.kernel_size[i],
                            strides=1,
                            padding='same',
                            dilation_rate=self.configs.dilation_rate[i],
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            activation=tf.nn.relu,
                            name='conv1d_word_{}'.format(i),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                self.configs.l2_regs)
                        )
                        if self.configs.use_residual:
                            if self.configs.concat_residual:
                                outputs = array_ops.concat(
                                    [outputs, temp], axis=-1)
                            else:
                                outputs = tf.layers.conv1d(
                                    outputs,
                                    filters=temp.get_shape()[-1],
                                    kernel_size=1,
                                    strides=1,
                                    padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=None,
                                    name='conv1d_res_{}'.format(i),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                        self.configs.l2_regs)
                                ) + temp
                        final_size = outputs.get_shape()[-1]
                        if self.mode == 'train':
                            outputs = tf.nn.dropout(outputs, self.dropout)
                # output_list.append(outputs)
            # outputs = tf.concat([outputs, self.word_embedding], axis=-1)

        # outputs = tf.concat(outputs, axis=-1)
        if self.configs.final_layer == 'cnn':
            self.logits = outputs
            self.configs.final_layer_filters[-1] = self.configs.num_classes
            for filters, size in zip(self.configs.final_layer_filters, self.configs.final_layer_kernel):
                self.logits = tf.layers.conv1d(
                    inputs=self.logits,
                    filters=filters,
                    kernel_size=size,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.nn.relu,
                    name='conv_final_%d_%d' % (filters, size),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                        self.configs.l2_regs)
                )
        else:
            W0 = tf.get_variable(
                'W0', shape=[final_size, self.configs.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b0 = tf.Variable(
                tf.zeros(self.configs.num_classes, dtype=tf.float32), name='b0'
            )
            l2_loss = tf.nn.l2_loss(W0) + tf.nn.l2_loss(b0)
            self.loss += l2_loss * self.configs.l2_regs
            nsteps = tf.shape(outputs)[1]
            outputs = tf.reshape(outputs, [-1, final_size])
            scores = tf.matmul(outputs, W0) + b0
            self.logits = tf.reshape(
                scores, [-1, nsteps, self.configs.num_classes], name='logits'
            )

    def _add_loss_op(self):
        if self.configs.use_crf:
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs=self.logits,
                tag_indices=self.labels,
                sequence_lengths=self.sequence_length
            )
            self.transition_params = transition_params
            self.loss = tf.reduce_mean(-log_likelihood + self.loss)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_length)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses + self.loss)
        with tf.name_scope('train'):
            self.loss += tf.losses.get_regularization_loss()
            tf.summary.scalar('loss', self.loss)

    def _get_feed_dict(self,
                       sentences,
                       sentence_length,
                       words,
                       word_length,
                       labels=None,
                       dropout=1.0):
        feed_dict = {}
        feed_dict[self.word_ids] = sentences
        feed_dict[self.sequence_length] = sentence_length
        feed_dict[self.char_ids] = words
        feed_dict[self.word_length] = word_length
        feed_dict[self.dropout] = dropout
        if labels is not None:
            feed_dict[self.labels] = labels
        return feed_dict

    def predict_batch(self,
                      sentences,
                      sentence_lengths,
                      words,
                      word_lengths):
        fd = self._get_feed_dict(
            sentences, sentence_lengths, words, word_lengths
        )
        if self.configs.use_crf:
            decode_tags, best_scores = self.sess.run([
                self.decode_tags, self.best_scores
            ], feed_dict=fd)
            return decode_tags, best_scores
        else:
            labels_pred = self.sess.run(self.label_preds, feed_dict=fd)
            return labels_pred, [0]

    def train_dev_loop(self,
                       train_iter, dev_iter, eval_freq, num_epochs, early_stopping,
                       dropout):
        self._add_summary()
        next_batch = train_iter.get_next()
        if early_stopping > 0:
            smaller_count = 0
            best = 0
        for i in range(num_epochs):
            self.sess.run(train_iter.initializer)
            try:
                while True:
                    sentences, sentence_lengths, words, word_lengths, labels = self.sess.run(
                        next_batch
                    )
                    fd = self._get_feed_dict(
                        sentences, sentence_lengths, words, word_lengths, labels, dropout
                    )
                    run = self.sess.run(
                        [self.train_op, self.train_summaries,
                            self.loss, self.global_step], fd
                    )
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {}".format(
                        time_str, run[3], run[2]))
                    self.logger.info("Step {}, loss {}".format(run[3], run[2]))
                    self.train_summaries_writer.add_summary(run[1], run[3])
            except tf.errors.OutOfRangeError:
                if (i + 1) % eval_freq == 0:
                    self._save_model()
                    metrics = self.evaluate_step(dev_iter)
                    self.logger.info("Evaluation {}".format(str(metrics)))
                    if early_stopping > 0:
                        if metrics['total']['f1'] < best:
                            smaller_count += 1
                        else:
                            smaller_count = 0
                            best = metrics['total']['f1']
                        if smaller_count >= early_stopping:
                            break
        self._save_model()
        self._close_session()

    def evaluate_step(self,
                      dev_iter):
        self.sess.run(dev_iter.initializer)
        next_dev_batch = dev_iter.get_next()
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        main_tag = [x.split('-')[-1] for x in self.configs.vocab_tags]
        class_metrics = {
            x: {
                'correct_preds': 0.,
                'total_preds': 0.,
                'total_correct': 0.
            } for x in main_tag
        }
        try:
            while True:
                sentences, sentence_lengths, words, word_lengths, labels = self.sess.run(
                    next_dev_batch
                )
                pred_labels, _ = self.predict_batch(
                    sentences, sentence_lengths, words, word_lengths
                )
                for lab, lab_pred, length in zip(labels, pred_labels,
                                                 sentence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]

                    lab_chunks = set(get_chunks(lab, self.configs.vocab_tags))
                    lab_pred_chunks = set(get_chunks(lab_pred,
                                                     self.configs.vocab_tags))

                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)
                    for tag in main_tag:
                        tag_lab_chunks = set(
                            chunk for chunk in lab_chunks if chunk[0] == tag)
                        tag_lab_pred_chunks = set(
                            chunk for chunk in lab_pred_chunks if chunk[0] == tag)
                        class_metrics[tag]['correct_preds'] += len(
                            tag_lab_chunks & tag_lab_pred_chunks)
                        class_metrics[tag]['total_preds'] += len(
                            tag_lab_pred_chunks)
                        class_metrics[tag]['total_correct'] += len(
                            tag_lab_chunks)
        except tf.errors.OutOfRangeError:
            pass
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs),
        res = {}
        res['total'] = {"f1": 100 * f1,
                        "precision": 100 * p, "recall": 100 * r, "acc": acc}
        for tag, value in class_metrics.items():
            p = value['correct_preds'] / \
                value['total_preds'] if value['correct_preds'] > 0 else 0
            r = value['correct_preds'] / \
                value['total_correct'] if value['correct_preds'] > 0 else 0
            f1 = 2 * p * r / (p + r) if value['correct_preds'] > 0 else 0
            res[tag] = {"f1": 100*f1, "precision": 100*p, "recall": 100*r}
        return res


if __name__ == '__main__':
    model = load_model('trained_model/81/model-3400')
