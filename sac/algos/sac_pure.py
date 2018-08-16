import numpy as np
import tensorflow as tf

from rllab.misc import logger
from sac.misc.sampler import rollouts, SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.policies import UniformPolicy

from . import batch2


def check_reuse(a, b):
    assert set(a) == set(b)

# minimal shim to satisfy rllab's get_action() interface
class TempPolicy(object):
    def __init__(self, sess, ob_in, ac_out):
        self.sess = sess
        self.ob_in = ob_in
        self.ac_out = ac_out

    def get_action(self, obs):
        ac = self.sess.run(self.ac_out, feed_dict = {
            self.ob_in : obs[None,:],
        })
        return ac[0], None

# evaluate the policy
# TODO does this really need to be seperate? it's mainly just logging
def evaluate_policy(env, policy, sampler, max_len, n_episodes, render):

    paths = rollouts(env, policy, max_len, n_episodes)

    total_returns = [path['rewards'].sum() for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]

    logger.record_tabular('return-average', np.mean(total_returns))
    logger.record_tabular('return-min', np.min(total_returns))
    logger.record_tabular('return-max', np.max(total_returns))
    logger.record_tabular('return-std', np.std(total_returns))
    logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
    logger.record_tabular('episode-length-min', np.min(episode_lengths))
    logger.record_tabular('episode-length-max', np.max(episode_lengths))
    logger.record_tabular('episode-length-std', np.std(episode_lengths))

    #self._eval_env.log_diagnostics(paths)
    if render:
        env.render(paths)


def sac_learn(
    sess, env,
    epochs, minibatch, buf_size, init_explore_steps,
    lr, scale_reward, discount, tau):

    dim_ac = env.action_space.flat_dim
    dim_ob = env.observation_space.flat_dim

    # placeholders
    obs_ph = tf.placeholder(tf.float32, (None, dim_ob), "ob")
    nextob_ph = tf.placeholder(tf.float32, (None, dim_ob), "next_ob")
    ac_ph = tf.placeholder(tf.float32, (None, dim_ac), "ac")
    rew_ph = tf.placeholder(tf.float32, (None, ), "rew")

    # value function
    vf = batch2.MLP("myvf", obs_ph, (64, 64), 1, tf.nn.relu)

    # policy
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    policy = batch2.SquashedGaussianPolicy("sgpolicy",
        obs_ph, (64, 64), dim_ac, tf.nn.relu, reg=reg)
    log_pi = policy.logp(policy.raw_ac)

    # double q functions - these ones are used "on-policy" in the vf loss
    q_in = tf.concat([obs_ph, policy.ac], axis=1)
    qf1 = batch2.MLP("qf1", q_in, (64, 64), 1, tf.nn.relu)
    qf2 = batch2.MLP("qf2", q_in, (64, 64), 1, tf.nn.relu)
    qf_min = tf.minimum(qf1.out, qf2.out)

    # policy loss
    policy_kl_loss = tf.reduce_mean(log_pi - qf_min)
    pi_reg_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES, scope=policy.name)
    pi_reg_losses += [policy.reg_loss]
    policy_loss = policy_kl_loss + tf.reduce_sum(pi_reg_losses)

    # value function loss
    vf_loss = 0.5 * tf.reduce_mean((vf.out - tf.stop_gradient(qf_min - log_pi))**2)

    # same q functions, but for the off-policy TD training
    qtrain_in = tf.concat([obs_ph, ac_ph], axis=1)
    qf1_t = batch2.MLP("qf1", qtrain_in, (64, 64), 1, tf.nn.relu, reuse=True)
    qf2_t = batch2.MLP("qf2", qtrain_in, (64, 64), 1, tf.nn.relu, reuse=True)

    # target (slow-moving) vf, used to update Q functions
    with tf.variable_scope('target'):
        vf_TDtarget = batch2.MLP("vf_target", nextob_ph, (64, 64), 1, tf.nn.relu)

    # q fn TD-target & losses
    ys = tf.stop_gradient(scale_reward * rew_ph + discount * vf_TDtarget.out)
    TD_loss1 = 0.5 * tf.reduce_mean((ys - qf1_t.out)**2)
    TD_loss2 = 0.5 * tf.reduce_mean((ys - qf2_t.out)**2)


    # training ops
    policy_opt_op = tf.train.AdamOptimizer(lr).minimize(
        policy_loss, var_list=policy.get_params_internal())

    vf_opt_op = tf.train.AdamOptimizer(lr).minimize(
        vf_loss, var_list=vf.vars)

    qf1_opt_op = tf.train.AdamOptimizer(lr).minimize(
        TD_loss1, var_list=qf1.vars)

    qf2_opt_op = tf.train.AdamOptimizer(lr).minimize(
        TD_loss2, var_list=qf2.vars)

    train_ops = [policy_opt_op, vf_opt_op, qf1_opt_op, qf2_opt_op]

    # ops to update slow-moving target vf
    vf_target_moving_avg_ops = [
        tf.assign(target, (1 - tau) * target + tau * source)
        for target, source in zip(vf_TDtarget.vars, vf.vars)
    ]

    # do it
    sess.run(tf.global_variables_initializer())
    sess.run(vf_target_moving_avg_ops)

    # wrap policy for rllab interface
    policy_wrapper = TempPolicy(sess, obs_ph, policy.ac)
    policy_deterministic = TempPolicy(sess, obs_ph, tf.tanh(policy.mu))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)
    initial_exploration_done = False
    n_train_repeat = 1

    pool = SimpleReplayBuffer(env_spec=env.spec, max_replay_buffer_size=buf_size)
    SAMPLER_PARAMS = {
        'max_path_length': 1000,
        'min_pool_size': 1000,
        'batch_size': minibatch,
    }
    sampler = SimpleSampler(**SAMPLER_PARAMS)
    sampler.initialize(env, initial_exploration_policy, pool)

    for epoch in range(epochs):

        logger.push_prefix('Epoch #%d | ' % epoch)

        epoch_length = 1000
        for t in range(epoch_length):
            if not initial_exploration_done:
                if epoch_length * epoch >= init_explore_steps:
                    sampler.set_policy(policy_wrapper)
                    initial_exploration_done = True
            sampler.sample()
            if not sampler.batch_ready():
                continue

            for i in range(n_train_repeat):
                batch = sampler.random_batch()
                feed_dict = {
                    obs_ph: batch['observations'],
                    ac_ph: batch['actions'],
                    rew_ph: batch['rewards'],
                    nextob_ph: batch['next_observations'],
                }
                sess.run(train_ops, feed_dict)
                sess.run(vf_target_moving_avg_ops)

        eval_n_episodes = 1
        render = True
        evaluate_policy(env, policy_deterministic, sampler,
            sampler._max_path_length, eval_n_episodes, render)

        logger.record_tabular('epoch', epoch)
        sampler.log_diagnostics()
        logger.dump_tabular(with_prefix=False)
        logger.pop_prefix()

    sampler.terminate()

