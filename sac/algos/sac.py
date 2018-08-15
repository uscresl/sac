from numbers import Number

import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from .base import RLAlgorithm
from sac.misc import tf_utils
from sac.misc.sampler import rollouts, SimpleSampler
from sac.replay_buffers import SimpleReplayBuffer
from sac.policies import UniformPolicy
from sac.core.serializable import deep_clone

from . import batch2
james = True
james_policy = True

def check_reuse(a, b):
    assert set(a) == set(b)

# minimal shim to make policy wrapper of DiagGaussianPd
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

    # DEBUG why ?????
    with tf.variable_scope("low_level_policy", reuse=True):
        _eval_env = deep_clone(env)

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
        evaluate_policy(_eval_env, policy_deterministic, sampler,
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


class SAC(RLAlgorithm, Serializable):

    def __init__(
            self,
            base_kwargs,

            env,
            policy,
            initial_exploration_policy,
            qf1,
            qf2,
            vf,
            pool,
            plotter=None,

            lr=3e-3,
            scale_reward=1,
            discount=0.99,
            tau=0.01,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            save_full_state=False,
    ):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.

            env (`rllab.Env`): rllab environment object.
            policy: (`rllab.NNPolicy`): A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.

            qf1 (`valuefunction`): First Q-function approximator.
            qf2 (`valuefunction`): Second Q-function approximator. Usage of two
                Q-functions improves performance by reducing overestimation
                bias.
            vf (`ValueFunction`): Soft value function approximator.

            pool (`PoolBase`): Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.

            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.

            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
        """

        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        # all "in haarnoja" comments valid for half-cheetah only

        self._env = env
        if not james_policy:
            self._policy = policy
        # is gaussian in haarnoja
        self._initial_exploration_policy = initial_exploration_policy
        # is uniform in haarnoja
        #self._qf1 = qf1
        #self._qf2 = qf2
        #self._vf = vf
        self._pool = pool
        self._plotter = plotter


        self._policy_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        # all lr == 3e-4 in haarnoja
        self._scale_reward = scale_reward
        # == 5 for half-cheetah in haarnoja
        self._discount = discount
        # == .99 in haarnoja
        self._tau = tau
        # == .005 in haarnoja
        self._target_update_interval = target_update_interval
        # == 1 for half-cheetah in haarnoja
        self._action_prior = action_prior
        # == 'uniform' in haarnoja

        # Reparameterize parameter must match between the algorithm and the
        # policy actions are sampled from.
        #assert reparameterize == self._policy._reparameterize
        assert reparameterize == True
        self._reparameterize = reparameterize
        # == True in haarnoja

        self._save_full_state = save_full_state

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_target_ops()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        print("{} / {} variables uninitialized.".format(
            len(uninit_vars), len(tf.global_variables())))
        self._sess.run(tf.variables_initializer(uninit_vars))


    @overrides
    def train(self):
        assert james
        assert james_policy
        sess = tf_utils.get_default_session()
        ob_in = self._observations_ph
        ac_out = self._policy.ac
        policy_wrapper = TempPolicy(sess, ob_in, ac_out)
        self._train(self._env, policy_wrapper, self._initial_exploration_policy, self._pool)

    def train2(self):
        """Initiate training of the SAC instance."""
        if james_policy:
            sess = tf_utils.get_default_session()
            ob_in = self._observations_ph
            ac_out = self._policy.ac
            policy_wrapper = TempPolicy(sess, ob_in, ac_out)
            self._train(self._env, policy_wrapper, self._initial_exploration_policy, self._pool)
        else:
            self._train(self._env, self._policy, self._initial_exploration_policy, self._pool)

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_pl = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Do),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, self._Da),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    @property
    def scale_reward(self):
        if callable(self._scale_reward):
            return self._scale_reward(self._iteration_pl)
        elif isinstance(self._scale_reward, Number):
            return self._scale_reward

        raise ValueError(
            'scale_reward must be either callable or scalar')

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """

        if james:
            qtrain_in = tf.concat([self._observations_ph, self._actions_ph], axis=1)
            self._qf1_t = batch2.MLP("qf1", qtrain_in, (64, 64), 1, tf.nn.relu, reuse=True)
            self._qf2_t = batch2.MLP("qf2", qtrain_in, (64, 64), 1, tf.nn.relu, reuse=True)

        else:
            self._qf1_t = self._qf1.get_output_for(
                self._observations_ph, self._actions_ph, reuse=True)  # N
            self._qf2_t = self._qf2.get_output_for(
                self._observations_ph, self._actions_ph, reuse=True)  # N

        with tf.variable_scope('target'):
            vf_target = batch2.MLP("vf_target", self._next_observations_ph, (64, 64), 1, tf.nn.relu)
            self._vf_target_params = vf_target.vars
            vf_next_target_t = vf_target.out
            #vf_next_target_t = self._vf.get_output_for(self._next_observations_ph)  # N
            #self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            self.scale_reward * self._rewards_ph +
            (1 - self._terminals_ph) * self._discount * vf_next_target_t
        )  # N

        if james:
            self._td_loss1_t = 0.5 * tf.reduce_mean((ys - self._qf1_t.out)**2)
            self._td_loss2_t = 0.5 * tf.reduce_mean((ys - self._qf2_t.out)**2)
        else:
            self._td_loss1_t = 0.5 * tf.reduce_mean((ys - self._qf1_t)**2)
            self._td_loss2_t = 0.5 * tf.reduce_mean((ys - self._qf2_t)**2)

        qf1_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss1_t,
            #var_list=self._qf1.get_params_internal()
            var_list=self._qf1.vars
        )
        qf2_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss2_t,
            #var_list=self._qf2.get_params_internal()
            var_list=self._qf2.vars
        )

        self._training_ops.append(qf1_train_op)
        self._training_ops.append(qf2_train_op)

    def _init_actor_update(self):
        """Create minimization operations for policy and state value functions.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and value functions with gradient descent, and appends them to
        `self._training_ops` attribute.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        See Equations (8, 13) in [1], for further information
        of the value function and policy function update rules.
        """

        D_s = self._actions_ph.shape.as_list()[1]

        if james_policy:
            reg = tf.contrib.layers.l2_regularizer(1e-3)
            self._policy = batch2.SquashedGaussianPolicy("sgpolicy",
                self._observations_ph, (64, 64), D_s, tf.nn.relu, reg=reg)
            actions = self._policy.ac
            log_pi = self._policy.logp(self._policy.raw_ac)
        else:
            actions, log_pi = self._policy.actions_for(
                observations=self._observations_ph, with_log_pis=True)


        self._vf = batch2.MLP("my_vf", self._observations_ph, (64, 64), 1, tf.nn.relu)
        self._vf_t = self._vf.out
        self._vf_params = self._vf.vars
        #self._vf_t = self._vf.get_output_for(self._observations_ph, reuse=True)  # N
        #self._vf_params = self._vf.get_params_internal()


        if self._action_prior == 'normal':
            policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                loc=tf.zeros(D_s), scale_diag=tf.ones(D_s))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0


        if james:
            assert not hasattr(self, "_qf1")
            q_in = tf.concat([self._observations_ph, actions], axis=1)
            self._qf1 = batch2.MLP("qf1", q_in, (64, 64), 1, tf.nn.relu)
            self._qf2 = batch2.MLP("qf2", q_in, (64, 64), 1, tf.nn.relu)
            log_target1 = self._qf1.out
            min_log_target = tf.minimum(self._qf1.out, self._qf2.out)
        else:
            log_target1 = self._qf1.get_output_for(
                self._observations_ph, actions, reuse=True)  # N
            log_target2 = self._qf2.get_output_for(
                self._observations_ph, actions, reuse=True)  # N
            min_log_target = tf.minimum(log_target1, log_target2)

        if self._reparameterize:
            policy_kl_loss = tf.reduce_mean(log_pi - log_target1)
        else:
            policy_kl_loss = tf.reduce_mean(log_pi * tf.stop_gradient(
                log_pi - log_target1 + self._vf_t - policy_prior_log_probs))

        policy_regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES,
            scope=self._policy.name)
        if james_policy:
            policy_regularization_losses += [self._policy.reg_loss]
        policy_regularization_loss = tf.reduce_sum(
            policy_regularization_losses)

        self.policy_loss = (policy_kl_loss
                       + policy_regularization_loss)

        # We update the vf towards the min of two Q-functions in order to
        # reduce overestimation bias from function approximation error.
        self._vf_loss_t = 0.5 * tf.reduce_mean((
          self._vf_t
          - tf.stop_gradient(min_log_target - log_pi + policy_prior_log_probs)
        )**2)

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=self.policy_loss,
            var_list=self._policy.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_t,
            var_list=self._vf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)

    def _init_target_ops(self):
        """Create tensorflow operations for updating target value function."""

        source_params = self._vf_params
        target_params = self._vf_target_params

        self._target_ops = [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target_params, source_params)
        ]

    @overrides
    def _init_training(self, env, policy, pool):
        super(SAC, self)._init_training(env, policy, pool)
        self._sess.run(self._target_ops)

    @overrides
    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        self._sess.run(self._training_ops, feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._sess.run(self._target_ops)

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if iteration is not None:
            feed_dict[self._iteration_pl] = iteration

        return feed_dict

    @overrides
    def log_diagnostics(self, iteration, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        qf1, qf2, vf, td_loss1, td_loss2, pi_loss = self._sess.run(
            (self._qf1_t.out, self._qf2_t.out, self._vf_t, self._td_loss1_t, self._td_loss2_t, self.policy_loss), feed_dict)

        logger.record_tabular('qf1-avg', np.mean(qf1))
        logger.record_tabular('qf1-std', np.std(qf1))
        logger.record_tabular('qf2-avg', np.mean(qf1))
        logger.record_tabular('qf2-std', np.std(qf1))
        logger.record_tabular('mean-qf-diff', np.mean(np.abs(qf1-qf2)))
        logger.record_tabular('vf-avg', np.mean(vf))
        logger.record_tabular('vf-std', np.std(vf))
        logger.record_tabular('mean-sq-bellman-error1', td_loss1)
        logger.record_tabular('mean-sq-bellman-error2', td_loss2)
        logger.record_tabular('pi_loss', np.mean(pi_loss))

        #self._policy.log_diagnostics(iteration, batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            snapshot = {
                'epoch': epoch,
                'algo': self
            }
        else:
            snapshot = {
                #'epoch': epoch,
                #'policy': self._policy,
                #'qf1': self._qf1,
                #'qf2': self._qf2,
                #'vf': self._vf,
                #'env': self._env,
            }

        return snapshot

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'qf1-params': self._qf1.get_param_values(),
            'qf2-params': self._qf2.get_param_values(),
            'vf-params': self._vf.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._qf1.set_param_values(d['qf1-params'])
        self._qf2.set_param_values(d['qf2-params'])
        self._vf.set_param_values(d['vf-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
