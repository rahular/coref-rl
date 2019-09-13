#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import random

import numpy as np
import tensorflow as tf

import coref_model_finetune as cm
import util
import statistics as stat
import pickle as pkl

from tqdm import tqdm
from collections import deque
from wiki.reward import reward


def get_weights(name):
    with open('./logs/{}/weights.pkl'.format(name), 'rb') as f:
        return pkl.load(f)


def average_models(all_vars, is_conll):
    all_assign = []
    if is_conll:
        values1 = get_weights('text-us')
        values2 = get_weights('kg-us')
        values3 = get_weights('joint-us')
    else:
        values1 = get_weights('text-us-wikicoref')
        values2 = get_weights('kg-us-wikicoref')
        values3 = get_weights('joint-us-wikicoref')
    for var, val1, val2, val3 in zip(all_vars, values1, values2, values3):
        all_assign.append(tf.assign(var, (val1 + val2 + val3) / 3))
    return all_assign


def get_sents(pronouns, props, top_antecedents, top_antecedent_scores, top_span_starts, top_span_ends):
    pred_ants, sampled_indices = model.sample_predicted_antecedents(
        top_antecedents, top_antecedent_scores)
    clusters, _ = model.get_predicted_clusters(
        top_span_starts, top_span_ends, pred_ants)
    tokens = props[0]

    # this is a bit redundant but just making sure the tokenization is the same as demo.py
    sents = [[token for token in lst if token.strip()] for lst in tokens]

    words = util.flatten(sents)

    string_clusters = []
    for cluster in clusters:
        string_clusters.append([' '.join(words[m[0]:m[1]+1]) for m in cluster])
    print(string_clusters)

    resolved = [' '.join(sent) for sent in sents]
    resolved_copy = resolved.copy()

    for cluster in string_clusters:
        # if all items in cluster are pronouns, skip to next cluster
        cluster_set = set(cluster)
        if pronouns.intersection(cluster_set) == cluster_set:
            continue

        for mention in cluster:
            # otherwise get the first non pronoun as antecedent
            if mention not in pronouns:
                antecedent = mention
                break

        # TODO: is there a better way than looping again??
        for mention in cluster:
            # then replace mentions with antecedent
            if mention and antecedent and mention != antecedent:
                resolved = [sent.replace(
                    ' ' + mention + ' ',  ' ' + antecedent + ' ') for sent in resolved]

    resolved_sents = []
    for idx in range(len(resolved)):
        if resolved[idx] != resolved_copy[idx]:
            resolved_sents.append(resolved[idx])

    return resolved_sents, list(sampled_indices)


def get_input(config, model):
    # print('Preparing inputs...')
    with open(config["train_path"]) as f:
        train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
    # train_examples = [model.tensorize_example(example, is_training=True) for example in tqdm(train_examples)]
    random.shuffle(train_examples)
    while True:
        for example in train_examples:
            example = model.tensorize_example(example, is_training=True)
            feed_dict = dict(zip(model.input_tensors, example))
            yield feed_dict
        random.shuffle(train_examples)


if __name__ == "__main__":
    pronouns = set(['mine', 'us', 'you', 'him', 'whoever', 'whomever', 'themselves', 'there', 'your', 'these', 'where', 'myself', 'whose', 'someone',
                    'ourselves', 'his', 'whichever', 'everybody', 'yourselves', 'anybody', 'which', 'our', 'herself', 'ours', 'ourselves', 'its', 'my', 'hers', 'their',
                    'her', 'whosever', 'whom', 'yourself', 'both', 'she', 'me', 'himself', 'itself', 'I', 'theirs', 'those', 'we', 'he', 'them', 'who', 'they',
                    'somebody', 'each other', 'something', 'it', 'yours', 'that', 'others', 'neither', 'none', 'wherever', 'some', 'thyself', 'no one', 'whereon',
                    'thy', 'whence', 'whereof', 'ye', 'theirself', 'whatever', 'whatnot', 'whether', 'thee', 'whosesoever', 'anyone', 'several', 'many', 'whereunto',
                    'ourself', 'thine', 'anything', 'such', 'any', 'all', 'aught', 'nobody', 'somewhat', 'either', 'whoso', 'themself', 'suchlike', 'whomsoever',
                    'whichsoever', 'wherewith', 'everything', 'idem', 'nothing', 'one another', 'this', 'wheresoever', 'as', 'most', 'whosoever', 'another',
                    'naught', 'thou', 'whereto', 'nought', 'one', 'other', 'theirselves', 'whatsoever', 'yon', 'whereby', 'whomso', 'everyone', 'enough',
                    'few', 'wherefrom', 'wherein', 'whereinto', 'wherewithal', 'yonder', 'what', 'ought', 'each'])

    config = util.initialize_from_env()
    loss_func = config['loss']

    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    # initialize reward module
    reward_model, ent_vocab, rel_vocab, corenlp_client = reward.init(
        config['reward_model'])

    log_dir = config['log_dir']
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    max_f1, patience_ctr, corenlp_ctr, eps = 0.0, 0, 0, 10**-32
    reward_history = deque(maxlen=100)
    patience_limit = config['patience']

    model = cm.CorefModel(config)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        accumulated_loss = 0.0

        ckpt = tf.train.get_checkpoint_state(log_dir)
        if config.distill:
            saver = tf.train.Saver()
            all_vars = tf.trainable_variables()
            all_assign = average_models(all_vars, config.is_conll)
            session.run(all_assign)
        elif ckpt and ckpt.model_checkpoint_path:
            print("Restoring the best model...")
            vars_to_restore = [
                v for v in tf.global_variables() if 'pg_reward' not in v.name]
            saver = tf.train.Saver(vars_to_restore)
            saver.restore(session, os.path.join(log_dir, "model.max.ckpt"))
        _, max_f1 = model.evaluate(session)

        initial_time = time.time()
        for feed_dict in get_input(config, model):
            tf_input_tensors, [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                               top_antecedents, top_antecedent_scores] = session.run([model.input_tensors, model.predictions],
                                                                                     feed_dict=feed_dict)

            reward_val = 0.0
            resolved_sents, sampled_indices = get_sents(pronouns, tf_input_tensors, top_antecedents,
                                                        top_antecedent_scores, top_span_starts, top_span_ends)

            reward_val = reward.get_reward(
                resolved_sents, reward_model, ent_vocab, rel_vocab, corenlp_client)
            if type(reward_val) == list:
                reward_val = sum(reward_val)

            reward_history.append(reward_val)
            if len(reward_history) > 1:
                reward_val = (reward_val - stat.mean(reward_history)
                              ) / (stat.stdev(reward_history) + eps)

            pg_reward = np.zeros_like(top_antecedent_scores)
            for i, j in enumerate(sampled_indices[::-1]):
                R = 0
                if j >= 0:
                    R = (reward_val * j) + 0.99 * R
                    pg_reward[i][j] = R

            feed_dict[model.pg_reward] = pg_reward + eps
            tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op],
                                                     feed_dict=feed_dict)
            accumulated_loss += tf_loss

            if tf_global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = tf_global_step / total_time

                average_loss = accumulated_loss / report_frequency
                print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step,
                                                                average_loss, steps_per_second))
                writer.add_summary(util.make_summary(
                    {"loss": average_loss}), tf_global_step)
                accumulated_loss = 0.0

            if tf_global_step % eval_frequency == 0:
                eval_frequency = report_frequency = np.random.randint(1, 11)
                saver.save(session, os.path.join(log_dir, "model"),
                           global_step=tf_global_step)
                try:
                    eval_summary, eval_f1 = model.evaluate(session)
                except:
                    # most time is spent here. so there is a high chance that
                    # the timeout exception from reward computation is caught here
                    eval_summary, eval_f1 = model.evaluate(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    util.copy_checkpoint(os.path.join(
                        log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr == patience_limit:
                        print("Patience ran out. Re-loading best model...")
                        saver.restore(session, os.path.join(
                            log_dir, "model.max.ckpt"))
                        patience_ctr = 0

                writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(util.make_summary(
                    {"max_eval_f1": max_f1}), tf_global_step)

                print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(
                    tf_global_step, eval_f1, max_f1))

            # periodically restart coreNLP server
            corenlp_ctr += 1
            if corenlp_ctr % 10**5 == 0:
                corenlp_client = reward.corenlp_restart(corenlp_client)

    # clean up reward module assets
    reward.corenlp_stop(corenlp_client)
