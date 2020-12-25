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
"""
BERT finetuning runner.

Modified by @mrzjy

Basically it's BERT encoder + Transformer decoder
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random

import modeling
import optimization
import tokenization
import tensorflow as tf

from eval_utils import get_eval_metrics
from transformer_decoder import TransformerDecoder

INF = 100000.0

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", "config/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "config/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("train_file", None, "train file")

flags.DEFINE_string("test_file", None, "test file")

flags.DEFINE_string("predict_file", None, "predict file")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_input_length", 128, "max_input_length")

flags.DEFINE_integer("max_target_length", 16, "max_target_length")

flags.DEFINE_integer("n_layers", None, "n layers, must be smaller than config layer")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("log_frequency", 200, "log_frequency")

flags.DEFINE_integer("batch_size", 128, "batch_size")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("epochs", 1.0, "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("beam_size", 20, "beam_size")
flags.DEFINE_float("length_penalty_weight", 0.8, "length_penalty_weight")
flags.DEFINE_bool("sampling", False, "Whether to use sampling")
flags.DEFINE_float("top_p", 0.25, "top_p sampling")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, label, x, y=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.label = label
        self.x = x
        self.y = y


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, insert_id=None,
                 target_ids=None, target_mask=None, **kwargs):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.insert_id = insert_id
        self.target_ids = target_ids
        self.target_mask = target_mask

        for k in kwargs:
            setattr(self, k, kwargs[k])


class DecoratorDataLoader:
    def __init__(self, task_name="text_generation"):
        self.num_train_samples = 0

    def convert_to_unicode(self, string_or_list):
        if isinstance(string_or_list, str):
            return tokenization.convert_to_unicode(string_or_list)
        elif isinstance(string_or_list, list):
            return [tokenization.convert_to_unicode(s) for s in string_or_list]
        else:
            raise Exception

    def get_train_examples(self, file, is_training=True):
        assert os.path.exists(file)
        examples = []
        with open(file, "r", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if i % 50000 == 0:
                    print("reading", i, len(examples), "examples")

                splits = l.strip().split("\t")
                if len(splits) != 2:
                    continue

                sentence = splits[1]
                for simile in splits[0].split(" || "):
                    example = InputExample(guid=i, label=1, x=sentence, y=simile)
                examples.append(example)

        if is_training:
            random.shuffle(examples)
        return examples

    def get_dev_examples(self, file):
        return self.get_train_examples(file, is_training=False)

    def get_test_examples(self, file):
        return self.get_train_examples(file, is_training=False)

    def get_predict_examples(self, file):
        assert os.path.exists(file)
        examples = []
        with open(file, "r", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if i % 50000 == 0:
                    print("reading", i, len(examples), "examples")

                splits = l.strip().split("\t")
                if len(splits) == 2:  # reading a test file (simile & context)
                    sentence = splits[1]
                    for simile in splits[0].split(" || "):
                        example = InputExample(guid=i, label=1, x=sentence, y=simile)
                elif len(splits) == 1:  # given only plain contexts
                    example = InputExample(guid=i, label=1, x=splits[0], y=None)
                else:
                    continue
                examples.append(example)
        return examples


def find_sublist_in_list(l, sub_l):
    for i in range(len(l) - len(sub_l)):
        if sub_l == l[i:i + len(sub_l)]:
            return i
    return 0  # pointing to [CLS]


marks = {"，", "。", "！", "？"}


def convert_single_example(ex_index, example, max_input_length, max_target_length, tokenizer, is_training=False):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    def tokenize(string_or_list):
        if isinstance(string_or_list, str):
            return tokenizer.tokenize(string_or_list)
        elif isinstance(string_or_list, list):
            tokens = []
            for s in string_or_list:
                tokens.extend(tokenizer.tokenize(s))
                tokens.append("[SEP]")
            return tokens[:-1]  # last [SEP] will be added outside
        else:
            raise Exception("Unknown type of input")

    try:
        assert len(example.x)
    except AssertionError:
        return None

    def truncate_length(ids, max_length):
        mask = [1] * len(ids)
        if len(ids) > max_length:
            ids = ids[-max_length:]
            mask = mask[-max_length:]
        while len(ids) < max_length:
            ids.append(0)
            mask.append(0)
        return ids, mask

    tokens_x = ["[CLS]"] + tokenize(example.x) + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens_x)
    input_ids, input_mask = truncate_length(input_ids, max_input_length)
    assert len(input_ids) == len(input_mask)

    if example.y is not None:
        # get simile
        tokens_y = tokenize(example.y)
        # get simile position
        insert_id = find_sublist_in_list(tokens_x, tokens_y)
        # remove redundent marks
        offset = 0 if tokens_x[insert_id] not in marks and tokens_x[insert_id + len(tokens_y)] not in marks else 1
        # get plain context (by removing its simile)
        tokens_x = tokens_x[:insert_id] + tokens_x[insert_id + len(tokens_y) + offset:]
        if tokens_x[-1] == "，":
            tokens_x = tokens_x[:-1]
        # refine input
        input_ids = tokenizer.convert_tokens_to_ids(tokens_x)
        input_ids, input_mask = truncate_length(input_ids, max_input_length)
        assert len(input_ids) == len(input_mask)

        target_ids = tokenizer.convert_tokens_to_ids(tokens_y + ["[SEP]"])
        target_ids, target_mask = truncate_length(target_ids, max_target_length)
        assert len(target_ids) == len(target_mask)
    else:
        insert_id, tokens_y, target_ids, target_mask = None, [], [], []

    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % example.guid)
        tf.logging.info("input_tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_x]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        if example.y is not None:
            tf.logging.info("insert_id: {}".format(str(insert_id)))
            tf.logging.info("target_tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens_y]))
            tf.logging.info("target_ids: %s" % " ".join([str(x) for x in target_ids]))
            tf.logging.info("target_mask: %s" % " ".join([str(x) for x in target_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        insert_id=insert_id,
        target_ids=target_ids,
        target_mask=target_mask)
    return feature


def write_tfdataset(examples, tokenizer, max_input_length, max_target_length, output_file, is_training=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    writer = tf.python_io.TFRecordWriter(output_file)
    ex_index = 0
    num_invalid = 0
    feature_list = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d" % ex_index)

        feature = convert_single_example(
            ex_index, example, max_input_length, max_target_length, tokenizer, is_training)
        if feature is None:
            num_invalid += 1
            continue

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        if is_training:
            features["insert_id"] = create_int_feature([feature.insert_id])
            features["target_ids"] = create_int_feature(feature.target_ids)
            features["target_mask"] = create_int_feature(feature.target_mask)
        else:
            feature_list.append(feature)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    return ex_index + 1 - num_invalid, feature_list  # return total num of samples


def read_tfdataset(input_file, max_input_length, max_target_length, is_training, drop_remainder, is_shuffle=True):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_input_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_input_length], tf.int64),
    }
    if is_training:
        name_to_features["insert_id"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["target_ids"] = tf.FixedLenFeature([max_target_length], tf.int64)
        name_to_features["target_mask"] = tf.FixedLenFeature([max_target_length], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        # tf.Example only supports tf.int64, but the TPU only supports tf.int32. So cast all int64 t64int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if not os.path.exists(input_file):
            raise Exception("input_file {} not exists".format(input_file))

        d = tf.data.TFRecordDataset(input_file)
        if is_training and is_shuffle:
            d = d.repeat()  # repeated indefinitely
            d = d.shuffle(buffer_size=50000)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn


def _truncate_seq_pair(tokens_context, tokens_query, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_context) + len(tokens_query)
        if total_length <= max_length:
            break
        if len(tokens_context) > len(tokens_query):
            tokens_context.pop()
        else:
            tokens_query.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 num_predictions=1, use_one_hot_embeddings=False, n_layers=None, features=None):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
        default_name="bert",
        n_layers=n_layers
    )

    encoder_outputs = model.get_sequence_output()
    final_hidden_shape = modeling.get_shape_list(encoder_outputs, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    insert_weights = tf.get_variable(
        "cls/insert/output_weights", [num_predictions, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    insert_bias = tf.get_variable(
        "cls/insert/output_bias", [num_predictions], initializer=tf.zeros_initializer())

    project_weights = tf.get_variable(
        "proj/insert/output_weights", [hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    project_bias = tf.get_variable(
        "proj/insert/output_bias", [hidden_size], initializer=tf.zeros_initializer())

    with tf.variable_scope("insertion"):
        if is_training:
            # I.e., 0.1 dropout
            encoder_outputs = tf.nn.dropout(encoder_outputs, keep_prob=0.9)

        final_hidden_matrix = tf.reshape(encoder_outputs, [batch_size * seq_length, hidden_size])
        insert_logits = tf.matmul(final_hidden_matrix, insert_weights, transpose_b=True)
        insert_logits = tf.nn.bias_add(insert_logits, insert_bias)
        insert_logits = tf.reshape(insert_logits, [batch_size, seq_length])
        insert_logits += (1.0 - tf.cast(input_mask, tf.float32)) * -10000.0
        insert_id = tf.cast(tf.argmax(insert_logits, axis=-1), tf.int32)
        # for each sample i ([L, H]) of batch, choose the element at insert_id[i] position, which is of shape [H]
        # then for a batch, we get a tensor of shape [B, H], namely a batch of vectors
        batch_indices = tf.range(0, batch_size)[:, tf.newaxis]  # [batch_size, 1]
        if is_training:
            indices = tf.concat([batch_indices, features["insert_id"][:, tf.newaxis]], axis=1)  # [batch_size, 2]
        else:
            indices = tf.concat([batch_indices, insert_id[:, tf.newaxis]], axis=1)  # [batch_size, 2]
        insert_vector = tf.gather_nd(encoder_outputs, indices)
        insert_vector = tf.matmul(insert_vector, project_weights, transpose_b=True)
        insert_vector = tf.nn.bias_add(insert_vector, project_bias)[:, tf.newaxis, :]
    return insert_logits, insert_vector, insert_id, encoder_outputs, model.embedding_table


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def model_fn_builder(bert_config, num_predictions, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     PAD_id=0, EOS_id=102, n_layers=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = tf.ones_like(input_mask)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        result = {}

        (insert_logits, insert_vector, insert_id, encoder_outputs, embedding_table) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, num_predictions, features=features)

        result["insert_logits"] = insert_logits
        result["insert_id"] = insert_id

        if mode == tf.estimator.ModeKeys.TRAIN:
            true_insert_id = features["insert_id"]
            target_ids = features["target_ids"]

            decoder = TransformerDecoder(params, True, embedding_table, PAD_id, EOS_id)
            result = decoder(encoder_outputs, input_mask, insert_vector, target_ids)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint, n_layers=n_layers)
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print("  name = {}, shape = {}{}".format(var.name, var.shape, init_string))

            loss = optimization.cross_entropy_loss_smooth(
                target_ids, result["logits"], smoothing_factor=params["label_smoothing"],
                vocab_size=params["vocab_size"])
            loss *= tf.cast(features["target_mask"], tf.float32)
            ce_loss = tf.reduce_mean(loss)

            def compute_insertion_loss(logits, positions, mask):
                # 1 for valid positions and 0 for invalid positions (will add -inf to logits)
                seq_length = tf.shape(logits)[1]
                one_hot_positions = tf.one_hot(positions, depth=seq_length, dtype=tf.float32)  # [B, L]
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                return -tf.reduce_mean(tf.reduce_sum(
                    one_hot_positions * log_probs * mask, axis=-1))

            insert_loss = compute_insertion_loss(insert_logits,
                                                 true_insert_id,
                                                 tf.cast(input_mask, tf.float32))

            total_loss = (ce_loss + insert_loss) / 2.0

            with tf.name_scope("train"):
                tf.summary.scalar("ce_loss", ce_loss)
                tf.summary.scalar("insert_loss", insert_loss)
                tf.summary.scalar("total_loss", total_loss)

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime."""

                def begin(self):
                    self._step = -1

                def before_run(self, run_context):
                    self._step += 1
                    return tf.train.SessionRunArgs([ce_loss, insert_loss])  # Asks for loss value.

                def after_run(self, run_context, run_values):
                    if self._step % params["log_frequency"] == 0:
                        values = run_values.results
                        print("INFO:tensorflow:ce_loss:{:.5f}, insert_loss:{:.5f}".format(
                            values[0], values[1]
                        ))

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, training_hooks=[_LoggerHook()])

        elif mode == tf.estimator.ModeKeys.EVAL:
            target_ids = features["target_ids"]
            decoder = TransformerDecoder(params, False, embedding_table, PAD_id, EOS_id)
            result = decoder(encoder_outputs, input_mask, insert_vector, target_ids)
            loss = optimization.cross_entropy_loss_smooth(
                target_ids, result["logits"], smoothing_factor=0.0,
                vocab_size=params["vocab_size"])
            loss *= tf.cast(features["target_mask"], tf.float32)
            ce_loss = tf.reduce_mean(loss)
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=ce_loss, predictions={"insert_id": insert_id},
                eval_metric_ops=get_eval_metrics(result["logits"], target_ids, params))

        elif mode == tf.estimator.ModeKeys.PREDICT:
            decoder = TransformerDecoder(params, False, embedding_table, PAD_id, EOS_id)
            result = decoder(encoder_outputs, input_mask, insert_vector, vocab_mask=None)

            predictions = {
                "predicts": result["predicts"],
                "length": result["length"],
                "insert_id": insert_id,
                "score": result["scores"]
            }
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % mode)
        return output_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_input_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_input_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)
    processor = DecoratorDataLoader()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(visible_devices) > 1 and visible_devices[0] != "":
        devices = ["device:GPU:{}".format(i) for i in visible_devices]
        distribution = tf.contrib.distribute.MirroredStrategy(
            devices=devices, cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(None, num_packs=2))
    else:
        distribution = None

    train_examples = None
    num_train_steps = 0
    num_warmup_steps = 0
    if FLAGS.do_train:
        if not os.path.exists(os.path.join(FLAGS.output_dir, "train.tf_record")):
            train_examples = processor.get_train_examples(FLAGS.train_file)
            print("###length of total train_examples:", len(train_examples))
            n_train_samples = len(train_examples)
        else:
            with open(os.path.join(FLAGS.output_dir, "num_train_samples"), "r", encoding="utf-8") as f:
                processor.num_train_samples = [int(l.strip()) for l in f.readlines()][0]
            n_train_samples = processor.num_train_samples
        num_train_steps = int(n_train_samples / FLAGS.batch_size * FLAGS.epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    gpu_options = tf.GPUOptions(allow_growth=True)
    gpu_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                        session_config=gpu_config,
                                        save_checkpoints_steps=int(num_train_steps / FLAGS.epochs),
                                        keep_checkpoint_max=int(FLAGS.epochs))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_predictions=1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        n_layers=FLAGS.n_layers
    )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.output_dir,
        config=run_config,
        warm_start_from=None,
        params={**FLAGS.flag_values_dict(), **bert_config.to_dict()})

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        train_file_exists = os.path.exists(train_file)
        print("# train_file_exists:", train_file_exists, " ;train_file:", train_file)
        if not train_file_exists:
            num_train_samples, _ = write_tfdataset(train_examples, tokenizer, FLAGS.max_input_length,
                                                   FLAGS.max_target_length, train_file, is_training=True)
            with open(os.path.join(FLAGS.output_dir, "num_train_samples"), "w", encoding="utf-8") as f:
                print(str(num_train_samples), file=f)
            processor.num_train_samples = num_train_samples
        else:
            with open(os.path.join(FLAGS.output_dir, "num_train_samples"), "r", encoding="utf-8") as f:
                processor.num_train_samples = [int(l.strip()) for l in f.readlines()][0]
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", processor.num_train_samples)
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = read_tfdataset(train_file, FLAGS.max_input_length,
                                        FLAGS.max_target_length,
                                        is_training=True,
                                        drop_remainder=True, )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_train_examples(FLAGS.test_file)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        write_tfdataset(eval_examples, tokenizer, FLAGS.max_input_length,
                        FLAGS.max_target_length, eval_file, is_training=True)
        tf.logging.info("***** Running eval *****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)

        eval_input_fn = read_tfdataset(eval_file, FLAGS.max_input_length,
                                       FLAGS.max_target_length,
                                       is_training=True,
                                       is_shuffle=False,
                                       drop_remainder=False)
        estimator.evaluate(input_fn=eval_input_fn)

    if FLAGS.do_predict:
        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_predictions=1,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            n_layers=FLAGS.n_layers,
        )

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.output_dir,
            config=run_config,
            warm_start_from=None,
            params={**FLAGS.flag_values_dict(), **bert_config.to_dict()})

        pred_examples = processor.get_predict_examples(FLAGS.predict_file)
        pred_file = os.path.join(FLAGS.output_dir, "pred.tf_record")
        _, pred_features = write_tfdataset(pred_examples, tokenizer, FLAGS.max_input_length,
                                           FLAGS.max_target_length, pred_file, is_training=False)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Batch size = %d", FLAGS.batch_size)
        predict_input_fn = read_tfdataset(pred_file, FLAGS.max_input_length,
                                          FLAGS.max_target_length,
                                          is_training=False,
                                          is_shuffle=False,
                                          drop_remainder=False, )
        result = estimator.predict(input_fn=predict_input_fn)

        # define output files
        output_predict_file = "output/generations.txt"
        ground_predict_file = "output/groundtruth.txt"
        bleu_file_ref = "output/bleu_ref"
        bleu_file_out = "output/bleu_out"
        context_file_out = "output/contexts"
        f_bleu_ref = tf.gfile.GFile(bleu_file_ref, "w")
        f_bleu_out = tf.gfile.GFile(bleu_file_out, "w")
        context_out = tf.gfile.GFile(context_file_out, "w")
        ground_out = tf.gfile.GFile(ground_predict_file, "w")
        insert_accur = 0
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for i, prediction in enumerate(result):
                if i % 200 == 0:
                    print(i)
                input_ids = pred_features[i].input_ids
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

                if pred_examples[i].y is not None:
                    context_tokens = tokenizer.tokenize(pred_examples[i].x.replace(pred_examples[i].y, ""))
                else:
                    context_tokens = tokenizer.tokenize(pred_examples[i].x)

                output_ids = prediction["predicts"]
                length = prediction["length"]
                insert_id = prediction["insert_id"]
                insert_tokens = tokenizer.convert_ids_to_tokens(output_ids[:length - 1])
                insert_tokens = [t for t in insert_tokens if t not in {"[PAD]", "[SEP]", "[CLS]"}]

                left = "".join(input_tokens[1:insert_id])
                insertion = "".join(insert_tokens)
                right = "".join([t for t in input_tokens[insert_id:] if t not in {"[PAD]", "[SEP]", "[CLS]"}])
                print(left, insertion, right)
                f_bleu_out.write(" ".join(insert_tokens) + "\n")
                context_out.write(" ".join(context_tokens) + "\n")

                if pred_examples[i].y is not None:
                    target_tokens = tokenizer.tokenize(pred_examples[i].y)
                    true_insert_id = pred_features[i].insert_id
                    insert_accur += int(insert_id == true_insert_id)
                    writer.write("\t".join([left, insertion, right, str(prediction["score"])]) + "\n")

                    ground_left = "".join(input_tokens[1:true_insert_id])
                    ground_insertion = "".join(target_tokens)
                    ground_right = "".join(
                        [t for t in input_tokens[true_insert_id:] if t not in {"[PAD]", "[SEP]", "[CLS]"}])

                    ground_out.write(
                        "\t".join([ground_left, ground_insertion, ground_right, str(prediction["score"])]) + "\n")
                    f_bleu_ref.write(" ".join(target_tokens) + "\n")

            print(insert_accur / (i + 1))


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
