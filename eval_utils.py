#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:44:40 2019

@author: transformer
Code Modified by @mrzjy
"""
import pickle

import jieba
import numpy as np
import tensorflow as tf

import collections
import math

import six


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the results have the same length (second dimension)."""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

        max_length = tf.maximum(x_length, y_length)

        x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
        y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
        return x, y


def get_seq_length(seq_ids):
    padding = tf.to_float(tf.equal(seq_ids, 0))
    pad_len = tf.cast(tf.reduce_sum(padding, axis=1), dtype=tf.int32)
    seq_len = tf.shape(seq_ids)[1] - pad_len
    return seq_len


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
    with tf.name_scope("loss", values=[logits, labels]):
        # Calculate smoothing cross entropy
        with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
            confidence = 1.0 - smoothing
            low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
            soft_targets = tf.one_hot(
                tf.cast(labels, tf.int32),
                depth=vocab_size,
                on_value=confidence,
                off_value=low_confidence)
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)

            # Calculate the best (lowest) possible value of cross entropy, and
            # subtract from the cross entropy loss.
            normalizing_constant = -(
                    confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
                    low_confidence * tf.log(low_confidence + 1e-20))
            xentropy -= normalizing_constant

            labels_len = get_seq_length(labels)
            weights = tf.sequence_mask(labels_len, tf.shape(labels)[1], dtype=logits.dtype)
        return xentropy * weights, weights


def _convert_to_eval_metric(metric_fn):
    """Wrap a metric fn that returns scores and weights as an eval metric fn.

  The input metric_fn returns values for the current batch. The wrapper
  aggregates the return values collected over all of the batches evaluated.

  Args:
    metric_fn: function that returns scores and weights for the current batch's
      logits and predicted labels.

  Returns:
    function that aggregates the scores and weights from metric_fn.
  """

    def problem_metric_fn(*args):
        """Returns an aggregation of the metric_fn's returned values."""
        (scores, weights) = metric_fn(*args)

        # The tf.metrics.mean function assures correct aggregation.
        return tf.metrics.mean(scores, weights)

    return problem_metric_fn


def get_eval_metrics(logits, labels, params):
    """Return dictionary of model evaluation metrics."""
    metrics = {
        "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
        "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
            logits, labels),
        "accuracy_per_sequence": _convert_to_eval_metric(
            padded_sequence_accuracy)(logits, labels),
        "neg_log_perplexity": _convert_to_eval_metric(padded_neg_log_perplexity)(
            logits, labels, params["vocab_size"]),
        "approx_bleu_score": _convert_to_eval_metric(
            bleu_score)(logits, labels),
        "rouge_2_fscore": _convert_to_eval_metric(
            rouge_2_fscore)(logits, labels),
        "rouge_L_fscore": _convert_to_eval_metric(
            rouge_l_fscore)(logits, labels),
    }

    # Prefix each of the metric names with "metrics/". This allows the metric
    # graphs to display under the "metrics" category in TensorBoard.
    metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
    return metrics


def padded_accuracy(logits, labels):
    """Percentage of times that predictions matches labels on non-0s."""
    with tf.variable_scope("padded_accuracy", values=[logits, labels]):
        logits, labels = _pad_tensors_to_same_length(logits, labels)
        weights = tf.to_float(tf.not_equal(labels, 0))
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        padded_labels = tf.to_int32(labels)
        return tf.to_float(tf.equal(outputs, padded_labels)), weights


def padded_accuracy_topk(logits, labels, k):
    """Percentage of times that top-k predictions matches labels on non-0s."""
    with tf.variable_scope("padded_accuracy_topk", values=[logits, labels]):
        logits, labels = _pad_tensors_to_same_length(logits, labels)
        weights = tf.to_float(tf.not_equal(labels, 0))
        effective_k = tf.minimum(k, tf.shape(logits)[-1])
        _, outputs = tf.nn.top_k(logits, k=effective_k)
        outputs = tf.to_int32(outputs)
        padded_labels = tf.to_int32(labels)
        padded_labels = tf.expand_dims(padded_labels, axis=-1)
        padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
        same = tf.to_float(tf.equal(outputs, padded_labels))
        same_topk = tf.reduce_sum(same, axis=-1)
        return same_topk, weights


def padded_accuracy_top5(logits, labels):
    return padded_accuracy_topk(logits, labels, 5)


def padded_sequence_accuracy(logits, labels):
    """Percentage of times that predictions matches labels everywhere (non-0)."""
    with tf.variable_scope("padded_sequence_accuracy", values=[logits, labels]):
        logits, labels = _pad_tensors_to_same_length(logits, labels)
        weights = tf.to_float(tf.not_equal(labels, 0))
        outputs = tf.to_int32(tf.argmax(logits, axis=-1))
        padded_labels = tf.to_int32(labels)
        not_correct = tf.to_float(tf.not_equal(outputs, padded_labels)) * weights
        axis = list(range(1, len(outputs.get_shape())))
        correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
        return correct_seq, tf.constant(1.0)


def padded_neg_log_perplexity(logits, labels, vocab_size):
    """Average log-perplexity excluding padding 0s. No smoothing."""
    num, den = padded_cross_entropy_loss(logits, labels, 0, vocab_size)
    return -num, den


def bleu_score(logits, labels):
    """Approximate BLEU score computation between labels and predictions.

  An approximate BLEU scoring method since we do not glue word pieces or
  decode the ids and tokenize the output. By default, we use ngram order of 4
  and use brevity penalty. Also, this does not have beam search.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch-size, length_labels]

  Returns:
    bleu: int, approx bleu score
  """
    predictions = tf.to_int32(tf.argmax(logits, axis=-1))
    bleu = tf.py_func(compute_bleu, (labels, predictions), tf.float32)
    return bleu, tf.constant(1.0)


def _get_ngrams_with_counter(segment, max_order):
    """Extracts all n-grams up to a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    use_bp: boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[
                ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in range(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)


def rouge_2_fscore(logits, labels):
    """ROUGE-2 F1 score computation between labels and predictions.

  This is an approximate ROUGE scoring method since we do not glue word pieces
  or decode the ids and tokenize the output.

  Args:
    logits: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    rouge2_fscore: approx rouge-2 f1 score.
  """
    predictions = tf.to_int32(tf.argmax(logits, axis=-1))
    rouge_2_f_score = tf.py_func(rouge_n, (predictions, labels), tf.float32)
    return rouge_2_f_score, tf.constant(1.0)


def _get_ngrams(n, text):
    """Calculates n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def rouge_n(eval_sentences, ref_sentences, n=2):
    """Computes ROUGE-N f1 score of two text collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Args:
    eval_sentences: Predicted sentences.
    ref_sentences: Sentences from the reference set
    n: Size of ngram.  Defaults to 2.

  Returns:
    f1 score for ROUGE-N
  """
    f1_scores = []
    for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
        eval_ngrams = _get_ngrams(n, eval_sentence)
        ref_ngrams = _get_ngrams(n, ref_sentence)
        ref_count = len(ref_ngrams)
        eval_count = len(eval_ngrams)

        # Count the overlapping ngrams between evaluated and reference
        overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
        overlapping_count = len(overlapping_ngrams)

        # Handle edge case. This isn't mathematically correct, but it's good enough
        if eval_count == 0:
            precision = 0.0
        else:
            precision = float(overlapping_count) / eval_count
        if ref_count == 0:
            recall = 0.0
        else:
            recall = float(overlapping_count) / ref_count
        f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

    # return overlapping_count / reference_count
    return np.mean(f1_scores, dtype=np.float32)


def rouge_l_fscore(predictions, labels):
    """ROUGE scores computation between labels and predictions.

  This is an approximate ROUGE scoring method since we do not glue word pieces
  or decode the ids and tokenize the output.

  Args:
    predictions: tensor, model predictions
    labels: tensor, gold output.

  Returns:
    rouge_l_fscore: approx rouge-l f1 score.
  """
    outputs = tf.to_int32(tf.argmax(predictions, axis=-1))
    rouge_l_f_score = tf.py_func(rouge_l_sentence_level, (outputs, labels),
                                 tf.float32)
    return rouge_l_f_score, tf.constant(1.0)


def rouge_l_sentence_level(eval_sentences, ref_sentences):
    """Computes ROUGE-L (sentence level) of two collections of sentences.

  Source: https://www.microsoft.com/en-us/research/publication/
  rouge-a-package-for-automatic-evaluation-of-summaries/

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    eval_sentences: The sentences that have been picked by the summarizer
    ref_sentences: The sentences from the reference set

  Returns:
    A float: F_lcs
  """

    f1_scores = []
    for eval_sentence, ref_sentence in zip(eval_sentences, ref_sentences):
        m = float(len(ref_sentence))
        n = float(len(eval_sentence))
        lcs = _len_lcs(eval_sentence, ref_sentence)
        f1_scores.append(_f_lcs(lcs, m, n))
    return np.mean(f1_scores, dtype=np.float32)


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.

  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.

  The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _f_lcs(llcs, m, n):
    """Computes the LCS-based F-measure score.

  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta ** 2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta ** 2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs


def extract_pretrained_embedding(embedding_file, vocab_file,
                                 embedding_dim=200, output_file="embedding.pkl"):
    w2v = {}
    # read vocab file
    print("reading {}".format(vocab_file))
    with open(vocab_file, "r") as f:
        vocab_set = set([l.strip() for l in f.readlines()])
    n_vocab_not_found = len(vocab_set)
    print("{} vocab to be found..".format(n_vocab_not_found))

    # read embedding file
    print("reading {}".format(embedding_file))
    n_lines_parsed = 0
    with open(embedding_file, "r", errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print("{} lines parsed, {} vocab left to be found".format(
                    n_lines_parsed, n_vocab_not_found))

            n_lines_parsed += 1

            if n_vocab_not_found == 0:
                break

            l = line.strip().split()
            if len(l) <= 2:
                print(line)
                continue

            try:
                token, embedding = l[0], [float(a) for a in l[1:]]
            except:
                continue
            if token not in vocab_set:
                continue
            if len(embedding) != embedding_dim:
                continue

            n_vocab_not_found -= 1
            w2v[token] = embedding

    print("{} lines parsed, {} vocab left to be found".format(
        n_lines_parsed, n_vocab_not_found))
    pickle.dump(w2v, open(output_file, "wb"))


def remove_unk_and_retokenize_jieba(tokens):
    return list(jieba.cut("".join([t for t in tokens if t != "[UNK]"])))


# Embedding Similarity
def greedy_match(fileone, filetwo, w2v):
    res1 = greedy_score(fileone, filetwo, w2v)
    res2 = greedy_score(filetwo, fileone, w2v)
    res_sum = (res1 + res2) / 2.0
    return np.mean(res_sum), 1.96 * np.std(res_sum) / float(len(res_sum)), np.std(res_sum)


def greedy_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()
    dim = 200  # embedding dimensions

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        tokens1 = remove_unk_and_retokenize_jieba(tokens1)
        tokens2 = remove_unk_and_retokenize_jieba(tokens2)

        X = np.zeros((dim,))
        y_count = 0
        x_count = 0
        o = 0.0
        Y = np.zeros((dim, 1))
        for tok in tokens2:
            if tok in w2v:
                vec = np.array(w2v[tok])
                Y = np.hstack((Y, (vec.reshape((dim, 1)))))
                y_count += 1

        for tok in tokens1:
            if tok in w2v:
                vec = np.array(w2v[tok])
                tmp = vec.reshape((1, dim)).dot(Y) / (np.linalg.norm(vec) * np.linalg.norm(Y))
                o += np.max(tmp)
                x_count += 1

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0)
            continue

        o /= float(x_count)
        scores.append(o)
    return np.asarray(scores)


def extrema_score(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        tokens1 = remove_unk_and_retokenize_jieba(tokens1)
        tokens2 = remove_unk_and_retokenize_jieba(tokens2)
        X = []
        for tok in tokens1:
            if tok in w2v:
                X.append(w2v[tok])
        Y = []
        for tok in tokens2:
            if tok in w2v:
                Y.append(w2v[tok])

        # if none of the words have embeddings in ground truth, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X, 0)  # get abs of min
        xtrema = []
        for i in range(len(xmax)):
            if np.abs(xmin[i]) > xmax[i]:
                xtrema.append(xmin[i])
            else:
                xtrema.append(xmax[i])
        X = np.array(xtrema)  # get extrema

        ymax = np.max(Y, 0)
        ymin = np.min(Y, 0)
        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T) / np.linalg.norm(X) / np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96 * np.std(scores) / float(len(scores)), np.std(scores)


def average(fileone, filetwo, w2v):
    f1 = open(fileone, 'r')
    f2 = open(filetwo, 'r')
    r1 = f1.readlines()
    r2 = f2.readlines()
    dim = 200  # dimension of embeddings

    scores = []

    for i in range(len(r1)):
        tokens1 = r1[i].strip().split(" ")
        tokens2 = r2[i].strip().split(" ")
        tokens1 = remove_unk_and_retokenize_jieba(tokens1)
        tokens2 = remove_unk_and_retokenize_jieba(tokens2)

        X = np.zeros((dim,))
        for tok in tokens1:
            if tok in w2v:
                X += w2v[tok]
        Y = np.zeros((dim,))
        for tok in tokens2:
            if tok in w2v:
                Y += w2v[tok]

        # if none of the words in ground truth have embeddings, skip
        if np.linalg.norm(X) < 0.00000000001:
            continue

        # if none of the words have embeddings in response, count result as zero
        if np.linalg.norm(Y) < 0.00000000001:
            scores.append(0)
            continue

        X = np.array(X) / np.linalg.norm(X)
        Y = np.array(Y) / np.linalg.norm(Y)
        o = np.dot(X, Y.T) / np.linalg.norm(X) / np.linalg.norm(Y)

        scores.append(o)

    scores = np.asarray(scores)
    return np.mean(scores), 1.96 * np.std(scores) / float(len(scores)), np.std(scores)
