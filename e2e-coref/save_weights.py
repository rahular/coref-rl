import os
import sys
import util
import pickle as pkl
import coref_model as cm
import tensorflow as tf

def load_model():
    config = util.initialize_from_env()
    log_dir = config['log_dir']
    model = cm.CorefModel(config)
    with tf.Session() as session:
        vars_to_restore = [v for v in tf.global_variables() if 'pg_reward' not in v.name]
        saver = tf.train.Saver(vars_to_restore)
        saver.restore(session, os.path.join(log_dir, "model.max.ckpt"))
        all_vars = tf.trainable_variables()
        values = session.run(all_vars)
        return values

if __name__ == '__main__':
    vals = load_model()
    with open('./logs/{}/weights.pkl'.format(sys.argv[1]), 'wb') as f:
        pkl.dump(vals, f, pkl.HIGHEST_PROTOCOL)
