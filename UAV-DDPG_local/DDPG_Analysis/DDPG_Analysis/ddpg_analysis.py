"""Full factorial DDPG hyperparameter sweep for UAV-DDPG.

Run from an isolated output directory. The source environment and normalization
are imported from the user's repository; its per-step output.txt logging is
disabled in-memory to avoid generating millions of diagnostic lines.
"""
import csv
import itertools
import math
import os
import random
import sys
import time
import warnings

import numpy as np

# TensorFlow 1.14 emits NumPy dtype-alias FutureWarnings with newer NumPy.
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# This script targets the original UAV-DDPG stack:
# Python 3.7 + TensorFlow 1.14.x + GAST 0.2.2.
# Do not use TensorFlow 2.x APIs in this file.
import tensorflow as tf

# Keep TensorFlow's repeated deprecation/AutoGraph notices out of the sweep
# output. The version check below still enforces the base algorithm's runtime.
tf.logging.set_verbosity(tf.logging.ERROR)

SOURCE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DDPG"))
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SOURCE)
from UAV_env import UAVEnv
from state_normalization import StateNormalization

# Match the TensorFlow version specified by the base implementation.
_tf_major_minor = tuple(int(x) for x in tf.__version__.split(".")[:2])
if _tf_major_minor != (1, 14):
    raise RuntimeError(
        "The base UAV-DDPG algorithm requires TensorFlow 1.14.x; "
        "found TensorFlow {}.".format(tf.__version__)
    )

LEVELS = {
    "episodes": [500, 1000, 1500],
    # "actor_lr": [0.1, 0.2, 0.3, 0.5, 0.01, 0.02, 0.03, 0.05, 0.001, 0.002, 0.003, 0.005, 0.0001, 0.0002, 0.0003, 0.0005],
    "actor_lr": [0.1, 0.2, 0.3, 0.5, 0.01, 0.02, 0.03, 0.05, 0.001, 0.002, 0.003, 0.005, 0.0001, 0.0002, 0.0003, 0.0005],
    "critic_lr": [0.1, 0.2, 0.3, 0.5, 0.01, 0.02, 0.03, 0.05, 0.001, 0.002, 0.003, 0.005, 0.0001, 0.0002, 0.0003, 0.0005],
    "batch_size": [2, 4, 8, 16, 32, 64, 128],
}
MEMORY_CAPACITY = 10000
GAMMA = 0.001
TAU = 0.01
NOISE_STD = 0.01
EVAL_EPISODES = 10
SEED = 1


def silent_reset2(self, delay, x, y, offloading_ratio, task_size, ue_id):
    """Mirror UAVEnv.reset2 while omitting its per-transition disk logging."""
    self.sum_task_size -= self.task_list[ue_id]
    for i in range(self.M):
        tmp = np.random.rand(2)
        theta_ue = tmp[0] * np.pi * 2
        dis_ue = tmp[1] * self.delta_t * self.v_ue
        self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
        self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
        self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
    self.reset_step()


UAVEnv.reset2 = silent_reset2


class Agent:
    def __init__(self, a_dim, s_dim, a_bound, actor_lr, critic_lr, batch_size):
        self.batch_size = batch_size
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.s_dim, self.a_dim, self.a_bound = s_dim, a_dim, a_bound
        self.S = tf.placeholder(tf.float32, [None, s_dim], "s")
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], "s_")
        self.R = tf.placeholder(tf.float32, [None, 1], "r")
        with tf.variable_scope("Actor"):
            self.a = self.build_actor(self.S, "eval", True)
            a_target = self.build_actor(self.S_, "target", False)
        with tf.variable_scope("Critic"):
            q = self.build_critic(self.S, self.a, "eval", True)
            q_target_net = self.build_critic(self.S_, a_target, "target", False)
        ae = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval")
        at = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target")
        ce = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/eval")
        ct = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Critic/target")
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(at + ct, ae + ce)]
        q_target = self.R + GAMMA * q_target_net
        td_loss = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.critic_train = tf.train.AdamOptimizer(critic_lr).minimize(td_loss, var_list=ce)
        actor_loss = -tf.reduce_mean(q)
        self.actor_train = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss, var_list=ae)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name="l1", trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name="l2", trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name="l3", trainable=trainable)
            out = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name="a", trainable=trainable)
            return tf.multiply(out, self.a_bound[1], name="scaled_a")

    def build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n = 400
            w_s = tf.get_variable("w1_s", [self.s_dim, n], trainable=trainable)
            w_a = tf.get_variable("w1_a", [self.a_dim, n], trainable=trainable)
            b = tf.get_variable("b1", [1, n], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w_s) + tf.matmul(a, w_a) + b)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name="l2", trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name="l3", trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)

    def choose_action(self, state):
        return self.sess.run(self.a, {self.S: state[np.newaxis, :]})[0]

    def store(self, s, a, r, s_next):
        idx = self.pointer % MEMORY_CAPACITY
        self.memory[idx] = np.hstack((s, a, [r], s_next))
        self.pointer += 1

    def learn(self):
        self.sess.run(self.soft_replace)
        idx = np.random.choice(MEMORY_CAPACITY, size=self.batch_size)
        batch = self.memory[idx]
        bs = batch[:, :self.s_dim]
        ba = batch[:, self.s_dim:self.s_dim + self.a_dim]
        br = batch[:, -self.s_dim - 1:-self.s_dim]
        bs_next = batch[:, -self.s_dim:]
        self.sess.run(self.actor_train, {self.S: bs})
        self.sess.run(self.critic_train, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_next})

    def close(self):
        self.sess.close()


def evaluate(agent, seed):
    np.random.seed(seed)
    random.seed(seed)
    env = UAVEnv()
    normalizer = StateNormalization()
    scores = []
    for _ in range(EVAL_EPISODES):
        state = env.reset()
        total = 0.0
        step = 0
        while step < env.slot_num:
            action = np.clip(agent.choose_action(normalizer.state_normal(state)), *env.action_bound)
            next_state, reward, terminal, redo, ratio_change, reset_dist = env.step(action)
            if redo:
                continue
            state = next_state
            total += reward
            step += 1
            if reset_dist:
                action[2] = -1
            if ratio_change:
                action[3] = -1
            if terminal:
                break
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def run_trial(episodes, actor_lr, critic_lr, batch_size, trial_no):
    tf.reset_default_graph()
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    env = UAVEnv()
    normalizer = StateNormalization()
    agent = Agent(env.action_dim, env.state_dim, env.action_bound,
                  actor_lr, critic_lr, batch_size)
    ep_returns = []
    max_steps = env.slot_num
    for ep in range(episodes):
        state = env.reset()
        total = 0.0
        j = 0
        while j < max_steps:
            action = agent.choose_action(normalizer.state_normal(state))
            action = np.clip(np.random.normal(action, NOISE_STD), *env.action_bound)
            next_state, reward, terminal, redo, ratio_change, reset_dist = env.step(action)
            if redo:
                continue
            if reset_dist:
                action[2] = -1
            if ratio_change:
                action[3] = -1
            agent.store(normalizer.state_normal(state), action, reward,
                        normalizer.state_normal(next_state))
            if agent.pointer > MEMORY_CAPACITY:
                agent.learn()
            state = next_state
            total += reward
            if j == max_steps - 1 or terminal:
                break
            j += 1
        ep_returns.append(total)
    mean_eval, std_eval = evaluate(agent, SEED)
    last_100 = float(np.mean(ep_returns[-min(100, len(ep_returns)):]))
    agent.close()
    return {
        "trial": trial_no, "episodes": episodes, "actor_lr": actor_lr,
        "critic_lr": critic_lr, "batch_size": batch_size,
        "eval_mean_return": mean_eval, "eval_std_return": std_eval,
        "last_100_train_return": last_100,
    }


def main():
    combinations = list(itertools.product(LEVELS["episodes"], LEVELS["actor_lr"],
                                          LEVELS["critic_lr"], LEVELS["batch_size"]))
    csv_path = os.path.join(OUT, "ddpg_grid_results.csv")
    fields = ["trial", "episodes", "actor_lr", "critic_lr", "batch_size",
              "eval_mean_return", "eval_std_return", "last_100_train_return"]
    started = time.time()
    print("Full factorial sweep: {} configurations".format(len(combinations)), flush=True)
    print("Levels: {}".format(LEVELS), flush=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, params in enumerate(combinations, 1):
            result = run_trial(*params, trial_no=i)
            writer.writerow(result)
            f.flush()
            elapsed = time.time() - started
            print("{}/{} {} score={:.3f} +/- {:.3f}; elapsed={:.1f} min".format(
                i, len(combinations), params, result["eval_mean_return"],
                result["eval_std_return"], elapsed / 60), flush=True)
    print("Results: {}".format(csv_path), flush=True)


if __name__ == "__main__":
    main()
