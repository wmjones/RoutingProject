from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
import collections
import tensorflow as tf         # should change so that i dont import everything
from Config import Config


class MaskWrapper(rnn_cell_impl.RNNCell):
    def __init__(self, cell, cell_is_attention=True):
        super(MaskWrapper, self).__init__()
        self._cell = cell
        self.cell_is_attention = cell_is_attention

    def call(self, inputs, state):
        if self.cell_is_attention:
            attention_state = tf.contrib.seq2seq.AttentionWrapperState(cell_state=state.cell_state,
                                                                       attention=state.attention,
                                                                       time=state.time,
                                                                       alignments=state.alignments,
                                                                       alignment_history=state.alignment_history,
                                                                       attention_state=state.attention_state)
            cell_output, main_cell_state = self._cell(inputs, attention_state)
            cell_output = cell_output - state.mask*Config.LOGIT_PENALTY
            sample_ids = tf.argmax(tf.nn.softmax(cell_output), axis=1, output_type=tf.int32)
            next_mask = state.mask + tf.one_hot(sample_ids, depth=Config.NUM_OF_CUSTOMERS+1, dtype=tf.float32)
            next_cell_state = MaskWrapperAttnState(
                cell_state=attention_state.cell_state,
                time=attention_state.time,
                attention=attention_state.attention,
                alignments=attention_state.alignments,
                alignment_history=attention_state.alignment_history,
                attention_state=attention_state.attention_state,
                mask=next_mask)
            return cell_output, next_cell_state
        else:
            cell_output, main_cell_state = self._cell(inputs, state.cell_state)
            cell_output = cell_output - state.mask*Config.LOGIT_PENALTY
            sample_ids = tf.argmax(tf.nn.softmax(cell_output), axis=1, output_type=tf.int32)
            next_mask = state.mask + tf.one_hot(sample_ids, depth=Config.NUM_OF_CUSTOMERS+1, dtype=tf.float32)
            next_cell_state = MaskWrapperState(
                cell_state=main_cell_state,
                time=state.time + 1,
                mask=next_mask)
            return cell_output, next_cell_state

    def zero_state(self, batch_size, dtype):
        if self.cell_is_attention:
            attention_state_zero = self._cell.zero_state(batch_size, dtype)
            # return self._cell.zero_state(batch_size, dtype), tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS+1]))
            return MaskWrapperAttnState(
                cell_state=attention_state_zero.cell_state,
                time=attention_state_zero.time,
                attention=attention_state_zero.attention,
                alignments=attention_state_zero.alignments,
                alignment_history=attention_state_zero.alignment_history,
                attention_state=attention_state_zero.attention_state,
                mask=tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS+1]))
        else:
            return MaskWrapperState(
                cell_state=self._cell.zero_state(batch_size, dtype),
                time=array_ops.zeros([], dtype=dtypes.int32),
                mask=tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS+1]))

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        # return self._cell.state_size
        if self.cell_is_attention:
            attention_state_size = self._cell.state_size
            return MaskWrapperAttnState(
                cell_state=attention_state_size.cell_state,
                time=attention_state_size.time,
                attention=attention_state_size.attention,
                alignments=attention_state_size.alignments,
                alignment_history=attention_state_size.alignment_history,
                attention_state=attention_state_size.attention_state,
                mask=attention_state_size.alignments)
        else:
            return MaskWrapperState(
                cell_state=self._cell.state_size,
                time=tensor_shape.TensorShape([]),
                mask=tensor_shape.TensorShape([None, Config.NUM_OF_CUSTOMERS+1]))


class MaskWrapperAttnState(collections.namedtuple("MaskWrapperAttnState",
                                                  ("cell_state", "attention", "time", "alignments",
                                                   "alignment_history", "mask", "attention_state"))):

    def clone(self, **kwargs):
        return super(MaskWrapperAttnState, self)._replace(**kwargs)


class MaskWrapperState(collections.namedtuple("MaskWrapperState",
                                              ("cell_state", "time", "mask"))):

    def clone(self, **kwargs):
        return super(MaskWrapperState, self)._replace(**kwargs)
