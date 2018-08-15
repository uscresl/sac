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

    print("inside sac_learn()!")
    print("lr:", lr)

    dim_ac = env.action_space.flat_dim
    dim_ob = env.observation_space.flat_dim

    # placeholders
    _observations_ph = tf.placeholder(tf.float32, (None, dim_ob), "ob")
    _next_observations_ph = tf.placeholder(tf.float32, (None, dim_ob), "next_ob")
    _actions_ph = tf.placeholder(tf.float32, (None, dim_ac), "ac")
    _rewards_ph = tf.placeholder(tf.float32, (None, ), "rew")

    # value function
    _vf = batch2.MLP("my_vf", _observations_ph, (64, 64), 1, tf.nn.relu)
    _vf_t = _vf.out
    _vf_params = _vf.vars

    # policy
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    _policy = batch2.SquashedGaussianPolicy("sgpolicy",
        _observations_ph, (64, 64), dim_ac, tf.nn.relu, reg=reg)
    actions = _policy.ac
    log_pi = _policy.logp(_policy.raw_ac)

    # double q functions
    q_in = tf.concat([_observations_ph, actions], axis=1)
    _qf1 = batch2.MLP("qf1", q_in, (64, 64), 1, tf.nn.relu)
    _qf2 = batch2.MLP("qf2", q_in, (64, 64), 1, tf.nn.relu)
    log_target1 = _qf1.out
    min_log_target = tf.minimum(_qf1.out, _qf2.out)

    # policy loss
    policy_kl_loss = tf.reduce_mean(log_pi - log_target1)
    pi_reg_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES, scope=_policy.name)
    print("reg losses:", pi_reg_losses)
    pi_reg_losses += [_policy.reg_loss]
    policy_loss = policy_kl_loss + tf.reduce_sum(pi_reg_losses)

    # value function loss
    _vf_loss_t = 0.5 * tf.reduce_mean((
      _vf_t
      - tf.stop_gradient(min_log_target - log_pi)
    )**2)

    qtrain_in = tf.concat([_observations_ph, _actions_ph], axis=1)
    _qf1_t = batch2.MLP("qf1", qtrain_in, (64, 64), 1, tf.nn.relu, reuse=True)
    _qf2_t = batch2.MLP("qf2", qtrain_in, (64, 64), 1, tf.nn.relu, reuse=True)
    check_reuse(_qf1_t.vars, _qf1.vars)
    check_reuse(_qf2_t.vars, _qf2.vars)

    # target (slow-moving) vf, used to update Q functions
    with tf.variable_scope('target'):
        vf_target = batch2.MLP("vf_target", _next_observations_ph, (64, 64), 1, tf.nn.relu)
        _vf_target_params = vf_target.vars
        vf_next_target_t = vf_target.out

    # q fn TD-target & losses
    ys = tf.stop_gradient(scale_reward * _rewards_ph + discount * vf_next_target_t)
    _td_loss1_t = 0.5 * tf.reduce_mean((ys - _qf1_t.out)**2)
    _td_loss2_t = 0.5 * tf.reduce_mean((ys - _qf2_t.out)**2)


    # training ops
    policy_train_op = tf.train.AdamOptimizer(lr).minimize(
        loss=policy_loss,
        var_list=_policy.get_params_internal()
    )
    vf_train_op = tf.train.AdamOptimizer(lr).minimize(
        loss=_vf_loss_t,
        var_list=_vf_params
    )
    qf1_train_op = tf.train.AdamOptimizer(lr).minimize(
        loss=_td_loss1_t,
        var_list=_qf1.vars
    )
    qf2_train_op = tf.train.AdamOptimizer(lr).minimize(
        loss=_td_loss2_t,
        var_list=_qf2.vars
    )

    _training_ops = [
        policy_train_op,
        vf_train_op,
        qf1_train_op,
        qf2_train_op,
    ]

    # ops to update slow-moving target vf
    source_params = _vf_params
    target_params = _vf_target_params
    _target_ops = [
        tf.assign(target, (1 - tau) * target + tau * source)
        for target, source in zip(target_params, source_params)
    ]

    #buf_size = int(1e6)
    #buf_dims = (dim_ob, dim_ac, 1, dim_ob)
    #_buffer = batch2.ReplayBuffer(buf_size, buf_dims)


    # do it
    sess.run(tf.global_variables_initializer())
    sess.run(_target_ops)

    # wrap policy for rllab interface
    policy_wrapper = TempPolicy(sess, _observations_ph, _policy.ac)
    policy_deterministic = TempPolicy(sess, _observations_ph, tf.tanh(_policy.mu))

    initial_exploration_policy = UniformPolicy(env_spec=env.spec)
    initial_exploration_done = False
    _n_train_repeat = 1

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

        _epoch_length = 1000
        for t in range(_epoch_length):
            # TODO.codeconsolidation: Add control interval to sampler
            if not initial_exploration_done:
                if _epoch_length * epoch >= init_explore_steps:
                    sampler.set_policy(policy_wrapper)
                    initial_exploration_done = True
            sampler.sample()
            if not sampler.batch_ready():
                continue

            for i in range(_n_train_repeat):

                batch = sampler.random_batch()

                feed_dict = {
                    _observations_ph: batch['observations'],
                    _actions_ph: batch['actions'],
                    _rewards_ph: batch['rewards'],
                    _next_observations_ph: batch['next_observations'],
                }

                sess.run(_training_ops, feed_dict)
                sess.run(_target_ops)

        eval_n_episodes = 1
        render = True
        evaluate_policy(env, policy_deterministic, sampler,
            sampler._max_path_length, eval_n_episodes, render)


        #iteration = epoch*_epoch_length
        #batch = sampler.random_batch()
        #log_diagnostics(iteration, batch)

        #params = get_snapshot(epoch)
        #logger.save_itr_params(epoch, params)
        #times_itrs = gt.get_times().stamps.itrs

        #eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
        #total_time = gt.get_times().total
        #logger.record_tabular('time-train', times_itrs['train'][-1])
        #logger.record_tabular('time-eval', eval_time)
        #logger.record_tabular('time-sample', times_itrs['sample'][-1])
        #logger.record_tabular('time-total', total_time)
        logger.record_tabular('epoch', epoch)

        sampler.log_diagnostics()

        logger.dump_tabular(with_prefix=False)
        logger.pop_prefix()

    sampler.terminate()

