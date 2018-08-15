import itertools
import os
import numpy as np
import tensorflow as tf


RENDER_EVERY = 10

def printstats(var, name):
    print("{}: mean={:3f}, std={:3f}, min={:3f}, max={:3f}".format(
        name, np.mean(var), np.std(var), np.min(var), np.max(var)))

# for fixed length episodes
# expects env to have ep_len member variable
def sysid_simple_generator(pi, env, stochastic, test=False, force_render=None):

    N = env.N
    dim = pi.dim
    horizon = env.ep_len

    pi.set_is_train(not test)

    # Initialize history arrays
    obs = np.zeros((horizon, N, dim.ob_concat))
    acs = np.zeros((horizon, N, dim.ac))
    if pi.flavor == "embed":
        embeds = np.zeros((horizon, N, dim.embed))
    elif pi.flavor == "extra":
        embeds = np.zeros((horizon, N, dim.sysid))
    else:
        embeds = np.zeros((horizon, N, 1))
    rews = np.zeros((horizon, N))
    vpreds = np.zeros((horizon, N))
    # rolling window, starting with zeros
    ob_trajs = np.zeros((horizon, N, dim.window, dim.ob))
    ac_trajs = np.zeros((horizon, N, dim.window, dim.ac))

    npr = env.np_random

    for episode in itertools.count():

        _, ob_std = pi.ob_mean_std()
        ob_std[dim.ob:] = 0

        # TODO could make it possible to include more than one reset in a batch
        # without also resampling SysIDs. But is it actually useful?
        env.sample_sysid()
        ob = env.reset()
        assert ob.shape == (N, dim.ob_concat)
        ob_trajs *= 0
        ac_trajs *= 0

        # touch / rm this file to toggle rendering
        render = (force_render if force_render is not None
            else os.path.exists("render"))

        for step in range(horizon):

            if render and episode % RENDER_EVERY == 0:
                env.render()

            ob += 0.03 * ob_std * npr.normal(size=ob.shape)
            obs[step,:,:] = ob

            if test:
                ac, vpred, embed = pi.act_traj(
                    stochastic, ob, ob_trajs[step], ac_trajs[step])
            else:
                ac, vpred, embed = pi.act(stochastic, ob)

            # epsilon-greedy exploration (TODO: pass in params)
            rand_acts = np.random.uniform(-1.0, 1.0, size=ac.shape)
            epsilon = np.random.uniform(size=ac.shape[0])
            greedy = epsilon < 0.4
            ac[greedy] = rand_acts[greedy]

            acs[step,:,:] = ac
            vpreds[step,:] = vpred
            embeds[step,:,:] = embed

            if step < horizon - 1:
                ob_trajs[step+1] = np.roll(ob_trajs[step], -1, axis=1)
                ac_trajs[step+1] = np.roll(ac_trajs[step], -1, axis=1)
                ob_trajs[step+1,:,-1,:] = ob[:,:dim.ob]
                ac_trajs[step+1,:,-1,:] = ac

            ob, rew, _, _ = env.step(ac)
            rews[step,:] = rew

        # Episode over.

        # in console we want to print the task reward only
        ep_rews = np.sum(rews, axis=0)

        # evaluate SysID errors and add to the main rewards.
        #sysids = obs[0,:,dim.ob:]
        #assert np.all((sysids[None,:,:] == obs[:,:,dim.ob:]).flat)
        #embed_trues = pi.sysid_to_embedded(sysids)
        embed_estimates = pi.estimate_sysid(
            ob_trajs.reshape((horizon * N, dim.window, dim.ob)),
            ac_trajs.reshape((horizon * N, dim.window, dim.ac)))
        embed_estimates = embed_estimates.reshape((horizon, N, -1))
        err2s = (embeds - embed_estimates) ** 2
        assert len(err2s.shape) == 3
        meanerr2s = np.mean(err2s, axis=-1)
        # apply the err2 for each window to *all* actions in that window
        sysid_loss = 0 * rews
        for i in range(horizon):
            begin = max(i - dim.window, 0)
            sysid_loss[begin:i,:] += meanerr2s[i,:]
        sysid_loss *= (pi.alpha_sysid / dim.window)

        total_rews = rews - sysid_loss
        # TODO keep these separate and let the RL algorithm reason about it?

        # yield the batch to the RL algorithm
        yield {
            "ob" : obs, "vpred" : vpreds, "ac" : acs,
            "rew" : total_rews, "task_rews" : rews, "sysid_loss" : sysid_loss,
            "ob_traj" : ob_trajs, "ac_traj" : ac_trajs,
            "embed_true" : embeds, "embed_estimate" : embed_estimates,
            "ep_rews" : ep_rews, "ep_lens" : horizon + 0 * ep_rews,
        }


def add_vtarg_and_adv(seg, gamma, lam):
    rew = seg["rew"]
    vpred = seg["vpred"]
    T, N = rew.shape
    # making the assumption that vpred is a smooth function of (non-sysid) state
    # and the error here is small
    # also assuming no special terminal rewards
    vpred = np.vstack((vpred, vpred[-1,:]))
    gaelam = np.zeros((T + 1, N))
    for t in reversed(range(T)):
        delta = rew[t] + gamma * vpred[t+1] - vpred[t]
        gaelam[t] = delta + gamma * lam * gaelam[t+1]
    vpred = vpred[:-1]
    gaelam = gaelam[:-1]
    seg["adv"] = gaelam
    seg["tdlamret"] = gaelam + vpred


# flattens arrays that are (horizon, N, ...) shape into (horizon * N, ...)
def seg_flatten_batches(seg):
    for s in ("ob", "ac", "task_rews", "ob_traj", "ac_traj", "embed_true", "adv", "tdlamret", "vpred"):
        sh = seg[s].shape
        newshape = [sh[0] * sh[1]] + list(sh[2:])
        seg[s] = np.reshape(seg[s], newshape)


class ReplayBuffer(object):
    def __init__(self, N, dims):
        N = int(N)
        self.bufs = tuple(np.zeros((N, d)) for d in dims)
        self.N = N
        self.size = 0
        self.cursor = 0

    def add(self, *args):
        if self.size < self.N:
            self.size += 1
        if self.cursor == 0:
            print("replay buffer roll over")
        for buf, item in zip(self.bufs, args):
            buf[self.cursor] = item
        self.cursor = (self.cursor + 1) % self.N

    def sample(self, np_random, batch_size):
        idx = np_random.randint(self.size, size=batch_size)
        returns = [buf[idx] for buf in self.bufs]
        return returns


class MLP(object):
    def __init__(self, name, input, hid_sizes, output_size, activation, reg=None, reuse=False):
        x = input
        with tf.variable_scope(name):
            for i, size in enumerate(hid_sizes):
                x = tf.layers.dense(x, size, activation=activation,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=reg, bias_regularizer=reg,
                    reuse=reuse, name="fc_{}".format(i))
            self.out = tf.layers.dense(x, output_size,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=reg, use_bias=False,
                name="fc_out", reuse=reuse)
            if output_size == 1:
                self.out = self.out[:,0]
            # TODO: seems circular, can we get this without using strings?
            scope_name = tf.get_variable_scope().name
            self.vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

# TODO is there a TF op for this?
def lerp(a, b, theta):
    return (1.0 - theta) * a + theta * b



class SquashedGaussianPolicy(object):
    def __init__(self, name, input, hid_sizes, output_size, activation, reg=None, reuse=False):

        self.mlp = MLP(name, input, hid_sizes, 2*output_size, activation, reg, reuse)
        mu, logstd = tf.split(self.mlp.out, 2, axis=1)
        self.pdf = tf.Distributions.Normal(loc=mu, scale=tf.exp(logstd))
        self.raw_ac = self.pdf.sample()
        self.ac = tf.tanh(self.pdf.sample())

    # actions should be non-raw, e.g. with tanh already applied
    def logp(self, actions):
        EPS = 1e-6
        log_p = self.pdf.log_prob(action)
        squash_correction = tf.reduce_sum(tf.log(1 - actions**2 + EPS), axis=1)
        return log_p - squash_correction
