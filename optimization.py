# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

import tensorflow as tf
import re

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops


def cross_entropy_loss_smooth(labels, logits, smoothing_factor, vocab_size):
    """Calculate cross entropy with label smoothing"""
    with tf.name_scope("loss"):
        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy"):
            confidence = 1.0 - smoothing_factor
            low_confidence = (1.0 - confidence) / (vocab_size - 1)
            soft_targets = tf.one_hot(
                labels,
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)
            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(
                    confidence * tf.math.log(confidence) + (vocab_size - 1) *
                    low_confidence * tf.math.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant
        return xentropy


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, distill=False):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    # optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-08, )
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    if not distill:
        tvars = tf.trainable_variables()
    else:
        tvars = [var for var in tf.trainable_variables() if "distill" in var.name]
        tf.logging.info("training for distillation...")
        for var in tvars:
            tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _prepare(self):
        self.learning_rate_t = ops.convert_to_tensor(
            self.learning_rate, name='learning_rate')
        self.weight_decay_rate_t = ops.convert_to_tensor(
            self.weight_decay_rate, name='weight_decay_rate')
        self.beta_1_t = ops.convert_to_tensor(self.beta_1, name='beta_1')
        self.beta_2_t = ops.convert_to_tensor(self.beta_2, name='beta_2')
        self.epsilon_t = ops.convert_to_tensor(self.epsilon, name='epsilon')

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, 'm', self._name)
            self._zeros_slot(v, 'v', self._name)

    def _apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _resource_apply_dense(self, grad, var):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        # Standard Adam update.
        next_m = (
                tf.multiply(beta_1_t, m) +
                tf.multiply(1.0 - beta_1_t, grad))
        next_v = (
                tf.multiply(beta_2_t, v) + tf.multiply(1.0 - beta_2_t,
                                                       tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        next_param = var - update_with_lr

        return control_flow_ops.group(*[var.assign(next_param),
                                        m.assign(next_m),
                                        v.assign(next_v)])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        learning_rate_t = math_ops.cast(
            self.learning_rate_t, var.dtype.base_dtype)
        beta_1_t = math_ops.cast(self.beta_1_t, var.dtype.base_dtype)
        beta_2_t = math_ops.cast(self.beta_2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self.epsilon_t, var.dtype.base_dtype)
        weight_decay_rate_t = math_ops.cast(
            self.weight_decay_rate_t, var.dtype.base_dtype)

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = state_ops.assign(m, m * beta_1_t,
                               use_locking=self._use_locking)

        m_scaled_g_values = grad * (1 - beta_1_t)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        update = m_t / (math_ops.sqrt(v_t) + epsilon_t)

        if self._do_use_weight_decay(var.name):
            update += weight_decay_rate_t * var

        update_with_lr = learning_rate_t * update

        var_update = state_ops.assign_sub(var,
                                          update_with_lr,
                                          use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(
                    x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class MultistepAdamOptimizer(tf.compat.v1.train.AdamOptimizer):
    """Adam with SGD updates every n steps with accumulated gradients."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adam", n=4):
        super(MultistepAdamOptimizer, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
            use_locking=use_locking, name=name)
        self._n = n  # Call Adam optimizer every n batches with accumulated grads
        self._n_t = None  # n as tensor

    def _create_slots(self, var_list):
        """Create slot variables for Adam with accumulated gradients."""
        super(MultistepAdamOptimizer, self)._create_slots(var_list)
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=0 if self._n == 1 else 1,
                                       name="iter",
                                       colocate_with=first_var)
        for v in var_list:
            self._zeros_slot(v, "grad_acc", self._name)

    def _get_iter_variable(self):
        graph = (
            None if tf.executing_eagerly() else tf.get_default_graph())
        return self._get_non_slot_variable("iter", graph=graph)

    def _prepare(self):
        super(MultistepAdamOptimizer, self)._prepare()
        self._n_t = tf.convert_to_tensor(self._n, name="n")

    def _apply_cond(self, apply_fn, grad, var, *args, **kwargs):
        """Apply conditionally if counter is zero."""
        grad_acc = self.get_slot(var, "grad_acc")

        def apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs):
            total_grad = (grad_acc + grad) / tf.cast(self._n_t, grad.dtype)
            adam_op = apply_fn(total_grad, var, *args, **kwargs)
            with tf.control_dependencies([adam_op]):
                grad_acc_to_zero_op = grad_acc.assign(tf.zeros_like(grad_acc),
                                                      use_locking=self._use_locking)
            return tf.group(adam_op, grad_acc_to_zero_op)

        def accumulate_gradient(grad_acc, grad):
            assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)
            return tf.group(assign_op)  # Strip return value

        return tf.cond(
            tf.equal(self._get_iter_variable(), 0),
            lambda: apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs),
            lambda: accumulate_gradient(grad_acc, grad))

    def _apply_dense(self, grad, var):
        return self._apply_cond(
            super(MultistepAdamOptimizer, self)._apply_dense, grad, var)

    def _resource_apply_dense(self, grad, var):
        return self._apply_cond(
            super(MultistepAdamOptimizer, self)._resource_apply_dense, grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self._apply_cond(
            super(MultistepAdamOptimizer, self)._apply_sparse_shared, grad, var,
            indices, scatter_add)

    def _apply_sparse(self, grad, var):
        # TODO(fstahlberg): Implement a sparse version
        tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
        dense_grad = tf.convert_to_tensor(grad)
        return self._apply_cond(
            super(MultistepAdamOptimizer, self)._apply_dense, dense_grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
        # Note that conversion to a dense Tensor handles duplicate `indices`
        # correctly (summing them). A real sparse implementation will probably want
        # to override _resource_apply_sparse instead so it gets them de-duplicated
        # automatically.
        dense_grad = tf.convert_to_tensor(
            tf.IndexedSlices(values=grad, indices=indices,
                             dense_shape=tf.shape(var)))
        return self._apply_cond(
            super(MultistepAdamOptimizer, self)._resource_apply_dense,
            dense_grad, var)

    def _finish(self, update_ops, name_scope):
        """Updates beta_power variables every n batches and incrs counter."""
        iter_ = self._get_iter_variable()
        beta1_power, beta2_power = self._get_beta_accumulators()
        with tf.control_dependencies(update_ops):
            with tf.colocate_with(iter_):
                def update_beta_op():
                    update_beta1 = beta1_power.assign(
                        beta1_power * self._beta1_t,
                        use_locking=self._use_locking)
                    update_beta2 = beta2_power.assign(
                        beta2_power * self._beta2_t,
                        use_locking=self._use_locking)
                    return tf.group(update_beta1, update_beta2)

                maybe_update_beta = tf.cond(
                    tf.equal(iter_, 0), update_beta_op, tf.no_op)
                with tf.control_dependencies([maybe_update_beta]):
                    update_iter = iter_.assign(tf.mod(iter_ + 1, self._n_t),
                                               use_locking=self._use_locking)
        return tf.group(
            *update_ops + [update_iter, maybe_update_beta], name=name_scope)
