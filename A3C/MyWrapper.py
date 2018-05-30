from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
import collections
import tensorflow as tf         # should change so that i dont import everything
from Config import Config


class MaskWrapper(rnn_cell_impl.RNNCell):
    def __init__(self, cell, cell_is_attention=True):
        super(MaskWrapper, self).__init__()
        self._cell = cell

    def call(self, inputs, state):
        cell_output, new_AttnState = self._cell(inputs, state.AttnState)
        if Config.DIRECTION == 5 or Config.DIRECTION == 6:
            cell_output = new_AttnState.alignments
        # cell_output = tf.Print(cell_output, [tf.reduce_min(cell_output)], summarize=19)
        cell_output = cell_output - state.mask*Config.LOGIT_PENALTY
        sample_ids = tf.argmax(cell_output, axis=1, output_type=tf.int32)
        # cell_output = Config.LOGIT_CLIP_SCALAR*tf.nn.tanh(cell_output)
        if Config.STOCHASTIC == 0:
            next_mask = state.mask + tf.one_hot(sample_ids, depth=Config.NUM_OF_CUSTOMERS, dtype=tf.float32)
        else:
            next_mask = state.mask
        next_cell_state = MaskWrapperAttnState(
            AttnState=new_AttnState,
            mask=next_mask)
        return cell_output, next_cell_state

    def zero_state(self, batch_size, dtype):
        attention_state_zero = self._cell.zero_state(batch_size, dtype)
        return MaskWrapperAttnState(
            AttnState=attention_state_zero,
            mask=tf.zeros([batch_size, Config.NUM_OF_CUSTOMERS]))

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        attention_state_size = self._cell.state_size
        return MaskWrapperAttnState(
            AttnState=attention_state_size,
            mask=attention_state_size.alignments)


class MaskWrapperAttnState(collections.namedtuple("MaskWrapperAttnState",
                                                  ("AttnState", "mask"))):

    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(MaskWrapperAttnState, self)._replace(**kwargs))
