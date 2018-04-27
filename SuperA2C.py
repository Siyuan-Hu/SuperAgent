#!/usr/bin/env python
import keras
import tensorflow as tf
import numpy as np
import gym
import sys
import copy
import argparse
from A2C_Continuous import Actor, Critic, A2C


class MultitaskNetwork(object):

    def __init__(self, source_environment_names, target_environment_name):
        source_networks = []
        for environment_name in source_environment_names:
            source_networks.append(QNetwork(environment_name, True))
        target_network = QNetwork(target_environment_name, False)

    def synchronize_network(self, w, b):
        for network in source_networks:
            network.set_weight(w, b)
        target_network.set_weight(w, b)


class StudentAgent(object):
    """docstring for StudentNetwork"""

    def __init__(self, num_observation, num_action, actor_lr, critic_lr, action_low, action_high):
        self.actor = Actor(num_observation=num_observation,
                           num_action=num_action,
                           lr=actor_lr,
                           action_low=action_low,
                           action_high=action_high)
        with self.actor.graph.as_default():
            self.target_mu = tf.placeholder(tf.float32, [None, 1])
            self.target_sigma = tf.placeholder(tf.float32, [None, 1])
            self.actor_loss = tf.reduce_mean(tf.square(tf.subtract(
                self.target_mu, self.actor.mu)) + tf.square(tf.subtract(self.target_sigma, self.actor.sigma)))
            self.actor_optimizer = tf.train.AdamOptimizer(
                actor_lr).minimize(self.actor_loss, name="actor_optimizer")
            self.actor.sess.run(tf.global_variables_initializer())

        self.critic = Critic(num_observation=num_observation,
                             lr=critic_lr)
        with self.critic.graph.as_default():
            self.target_v = tf.placeholder(tf.float32, [None, 1])
            self.critic_loss = tf.reduce_mean(
                tf.square(tf.subtract(self.target_v, self.critic.q_values)))
            self.critic_optimizer = tf.train.AdamOptimizer(
                critic_lr).minimize(self.critic_loss, name="critic_optimizer")
            self.critic.sess.run(tf.global_variables_initializer())

    def train(self, states, mu, sigma, v):
        self.train_actor(states, mu, sigma)
        self.train_critic(states, v)

    def train_actor(self, states, mu, sigma):
        self.actor_optimizer.run(session=self.actor.sess, feed_dict={self.actor.state_input: states,
                                                                     self.target_mu: mu,
                                                                     self.target_sigma: sigma})

    def train_critic(self, states, v):
        self.critic_optimizer.run(session=self.critic.sess, feed_dict={self.critic.state_input: states,
                                                                       self.target_v: v})

    def get_actor_loss(self, states, mu, sigma):
        return self.actor_loss.eval(session=self.actor.sess, feed_dict={self.actor.state_input: states,
                                                                        self.target_mu: mu,
                                                                        self.target_sigma: sigma})

    def get_weight(self):
        w_mu, b_mu, w_sigma, b_sigma = self.actor.sess.run(
            [self.actor.w_mu, self.actor.b_mu, self.actor.w_sigma, self.actor.b_sigma])

        w2, b2 = self.critic.sess.run([self.critic.w2, self.critic.b2])

        return w_mu, b_mu, w_sigma, b_sigma, w2, b2

    def set_weight(self, w_mu, b_mu, w_sigma, b_sigma, w2, b2):
        self.actor.sess.run([self.actor.w_mu.assign(w_mu),
                             self.actor.b_mu.assign(b_mu),
                             self.actor.w_sigma.assign(w_sigma),
                             self.actor.b_sigma.assign(b_sigma)])

        self.critic.sess.run([self.critic.w2.assign(w2),
                              self.critic.b2.assign(b2)])

class Replay_Memory():

    def __init__(self,
                 batch_size=4,
                 memory_size=50000,
                 burn_in=10000):
        import collections
        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is
        # as a list of transitions.

        self.batch_size = batch_size
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = collections.deque(maxlen=memory_size)

    def sample(self):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.

        current_memory_size = len(self.memory)
        sample_index = np.random.choice(current_memory_size,
                                        size=self.batch_size)
        samples = [self.memory[idx] for idx in sample_index]

        return np.array(samples)

    def append(self, transition):
        # Appends transition to the memory.

        self.memory.append(transition)


class ExpertAgent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self,
                 environment_name,
                 num_observation,
                 num_action,
                 actor_lr,
                 critic_lr,
                 action_low,
                 action_high,
                 logger,
                 model_dir,
                 model_step,
                 save_path='tmp',
                 render=False,
                 resume=0,
                 batch_size=64,
                 memory_size=50000,
                 burn_in=10000,
                 open_monitor=0,
                 frequency_update=1000,
                 frequency_sychronize=10):

        self.environment_name = environment_name
        self.logger = logger
        self.render = render
        # parameter of frequency
        self.frequency_update = frequency_update
        self.frequency_sychronize = frequency_sychronize
        # initilize the replay memory with burn in
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.burn_in = burn_in

        self.replay_memory = Replay_Memory(self.batch_size,
                                           self.memory_size,
                                           self.burn_in)

        self.actor_net = Actor(num_observation, num_action, actor_lr, action_low,
                               action_high, model=model_dir + "/actor/actor-" + str(model_step))
        self.critic_net = Critic(
            num_observation, critic_lr, model=model_dir + "/critic/critic-" + str(model_step))
        # use monitor to generate video

        # if (open_monitor):
        # 	video_save_path = self.save_path + "/video/"
        # 	self.env = gym.wrappers.Monitor(self.env,
        # 									video_save_path,
        # 									resume=True)

        self.teach_burn_in_memory()

    def get_next_state(self, action, env):
        # given the action, return next state, reward, done and info
        next_state, reward, done, info = env.step(action)
        return next_state, reward, done, info

    def get_target(self, state):

        return self.actor_net.get_action_set(state)

    def get_target_q(self, state):
        return self.critic_net.get_critics(state)

    def initialize_env(self, env):
        return env.reset()

    def teach(self):
        # every time call this func,
        # (1), it will append a new episode into the replay memory with
        #      every step's current_state and q_values
        # (2), return the sample batch of current_state and q_values

        batch = self.replay_memory.sample()

        batch_state_lst = []
        #batch_action_lst = []
        batch_mu_lst = []
        batch_sigma_lst = []
        batch_q_values_lst = []

        for tmp_state, tmp_mu, tmp_sigma, tmp_q_values in batch:
            batch_state_lst.append(tmp_state)
            batch_mu_lst.append(tmp_mu)
            batch_sigma_lst.append(tmp_sigma)
            batch_q_values_lst.append(tmp_q_values)

        return batch_state_lst, batch_mu_lst, batch_sigma_lst, batch_q_values_lst

    def teach_burn_in_memory(self):

        env = gym.make(self.environment_name)
        done = False
        current_state = self.initialize_env(env)

        for i in range(self.burn_in):
            q_values = self.get_target_q([current_state])

            action, mu, sigma = self.get_target([current_state])

            next_state, _, done, _ = self.get_next_state(action,
                                                         env)
            self.replay_memory.append((current_state,
                                       mu[0],
                                       sigma[0],
                                       q_values[0]))
            if done:
                current_state = self.initialize_env(env)
            else:
                current_state = next_state
        env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    num_update = 2000

    environment_name_lst = ["InvertedPendulum-v2", "InvertedDoublePendulum-v2"]
    model_dir_lst = ["./expert_mujoco/InvertedPendulum-v2",
                     "./expert_mujoco/InvertedDoublePendulum-v2"]
    model_step_lst = [1400, 16400]

    teacher_agent_lst = []
    student_network_lst = []
    num_env = len(environment_name_lst)
    frequency_report_loss = 100

    actor_lr = 1e-3
    critic_lr = 1e-3

    # initilze the teacher agent and student network
    for idx in range(num_env):
        env = gym.make(environment_name_lst[idx])
        num_action = env.action_space.shape[0]
        num_observation = env.observation_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low
        env.close()

        teacher_agent_lst.append(ExpertAgent(environment_name=environment_name_lst[idx],
                                             num_observation=num_observation,
                                             num_action=num_action,
                                             actor_lr=actor_lr,
                                             critic_lr=critic_lr,
                                             action_low=action_low,
                                             action_high=action_high,
                                             logger=logger,
                                             model_dir=model_dir_lst[idx],
                                             model_step=model_step_lst[idx],
                                             burn_in=10000,
                                             batch_size=10))
        student_network_lst.append(StudentAgent(
            num_observation, num_action, actor_lr, critic_lr, action_low, action_high))

    loss = 0
    for idx_update in range(num_update):
        for idx in range(num_env):
            teacher_agent = teacher_agent_lst[idx]
            student_network = student_network_lst[idx]

            batch_state_lst, batch_mu_lst, batch_sigma_lst, batch_q_values_lst = teacher_agent.teach()

            student_network.train(
                batch_state_lst, batch_mu_lst, batch_sigma_lst, batch_q_values_lst)

            # TODO
            # loss += student_network.get_actor_loss(batch_state_lst, batch_mu_lst, batch_sigma_lst)

            next_student_network_idx = (idx + 1) % num_env
            next_student_network = student_network_lst[
                next_student_network_idx]

            # Share weights
            w_mu, b_mu, w_sigma, b_sigma, w2, b2 = student_network.get_weight()
            next_student_network.set_weight(w_mu, b_mu, w_sigma, b_sigma, w2, b2)

            if ((idx_update % frequency_report_loss) == 0):
                # print(loss / frequency_report_loss)
                env = gym.make(environment_name_lst[idx])
                print(student_network.actor.test(env, idx_update))
                env.close()
                loss = 0


if __name__ == '__main__':
    main(sys.argv)
