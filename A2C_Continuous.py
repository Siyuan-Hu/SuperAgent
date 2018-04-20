import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ENVIROMENT = 'InvertedPendulum-v2'

class A2C(object):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self,
                 env,
                 model_config_path,
                 actor_lr,
                 critic_lr,
                 num_episodes,
                 N_step=20,
                 render=False,
                 discount_factor=1,
                 model_step = None):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - N_step: The value of N in N-step A2C.
        self.env = env
        self.N_step = N_step

        # # enviroment
        # num_action = env.action_space.n
        num_observation = env.observation_space.shape[0]
        self.num_episodes = num_episodes
        self.render = render
        self.discount_factor = discount_factor

        self.env = gym.make(ENVIROMENT)

        # model
        if model_step == None:
            self.actor_model = Actor(num_observation, actor_lr, env.action_space.low, env.action_space.high)
            self.critic_model = Critic(num_observation, critic_lr)
        else:
            self.load_models(num_observation, actor_lr, critic_lr, env.action_space.low, env.action_space.high, model_step)



    def train(self, gamma=1.0):
        # Trains the model on a single episode using A2C.
        file = open("log.txt", "w")

        max_reward = -500
        test_frequence = 100
        self.gamma_N_step = gamma ** self.N_step
        for i in range(self.num_episodes):
            states, actions, rewards = self.generate_episode(env=self.env,
                                                             render=self.render)
            R, G = self.episode_reward2G_Nstep(states=states, actions=actions, rewards=rewards, 
                gamma=gamma, N_step=self.N_step, discount_factor=self.discount_factor)
            self.actor_model.train(np.vstack(states), np.vstack(G), np.vstack(actions))
            self.critic_model.train(np.vstack(states), np.vstack(R))

            if (i % test_frequence == 0):
                reward, std = self.test(i)
                file.write(str(reward)+" "+str(std)+"\n")
                print(reward, std)
                if reward >= max_reward:
                    self.save_models(i)
                    max_reward = reward

        file.close()

    def test(self, epc_idx, render=False):
        log_dir = './log'
        name_mean = 'test10_reward'
        name_std = 'test10_std'
        num_test = 10
        total_array = np.zeros(num_test)
        for j in range(num_test):
            _, _, rs = self.generate_episode(self.env, render)
            total_array[j] = np.sum(rs)

        summary_var(log_dir, name_mean, np.mean(total_array), epc_idx)
        summary_var(log_dir, name_std, np.std(total_array), epc_idx)

        return np.mean(total_array), np.std(total_array)

    def generate_episode(self, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []

        state = env.reset()
        num_observation = env.observation_space.shape[0]
        while True:
            if render:
                env.render()
            action = self.actor_model.get_action(state)
            states.append(state)
            actions.append(action)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            if done:
                break

        return states, actions, rewards

    def episode_reward2G_Nstep(self, states, actions, rewards, gamma, N_step, discount_factor):
        ## TODO
        # how to get the output
        critic_output = self.critic_model.get_critics(states)
        num_total_step = len(rewards)
        # R: list, is the symbol "R_t" in the alorithm 2
        # G: list, is the difference between R and V(S_t)
        R = [None] * num_total_step
        G = [None] * num_total_step
        for t in range(num_total_step - 1, -1, -1):
            V_end = 0 if (t + N_step >= num_total_step) else critic_output[t + N_step]
            R[t] = (self.gamma_N_step) * V_end
            gamma_k = 1
            for k in range(N_step):
                R[t] += discount_factor * (gamma_k) * (rewards[t + k] if (t + k < num_total_step) else 0)
                gamma_k *= gamma
            G[t] = R[t] - critic_output[t][0]

        return R, G

    def save_models(self, model_step):
        self.actor_model.save_model(model_step)
        self.critic_model.save_model(model_step)

    def load_models(self, num_observation, actor_lr, critic_lr, action_low, action_high, model_step):
        self.actor_model = Actor(num_observation, actor_lr, action_low, action_high, model = "./models/actor/actor-"+str(model_step))
        self.critic_model = Critic(num_observation, critic_lr, model = "./models/critic/critic-"+str(model_step))

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=5000000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=1e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-3, help="The critic's learning rate.")
    parser.add_argument('--N_step', dest='N_step', type=int,
                        default=100, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()

class Actor(object):
    def __init__(self, num_observation, lr, action_low, action_high, model = None):
        self.num_observation = num_observation
        self.learning_rate = lr
        self.action_low = action_low
        self.action_high = action_high


        if model != None:
            self.load_model(model)
        else:
            self.graph = tf.Graph()
            self.sess = tf.Session(graph = self.graph)
            with self.graph.as_default():
                self.create_mlp()
                self.create_optimizer()
                self.sess.run(tf.global_variables_initializer())

    def train(self, states, G, actions):
        self.optimizer.run(session=self.sess, feed_dict={self.state_input: states, self.G: G, self.action: actions})

    def create_mlp(self):
        # Craete multilayer perceptron (one hidden layer with 20 units)
        self.hidden_units = 200

        self.G = tf.placeholder(tf.float32, [None, 1], name = 'G')
        self.action = tf.placeholder(tf.float32, [None, 1], name = 'action')

        self.w1 = self.create_weights([self.num_observation, self.hidden_units])
        self.b1 = self.create_bias([self.hidden_units])

        self.state_input = tf.placeholder(tf.float32, [None, self.num_observation], name = "state_input")

        h_layer = tf.nn.relu(tf.matmul(self.state_input, self.w1) + self.b1)

        self.w_mu = self.create_weights([self.hidden_units, 1])
        self.b_mu = self.create_bias([1])
        self.mu = tf.nn.tanh(tf.add(tf.matmul(h_layer, self.w_mu), self.b_mu), name = "mu")

        self.w_sigma = self.create_weights([self.hidden_units, 1])
        self.b_sigma = self.create_bias([1])
        self.sigma = tf.nn.softplus(tf.add(tf.matmul(h_layer, self.w_sigma), self.b_sigma), name = "sigma")

        self.mu, self.sigma = self.mu * self.action_high, self.sigma + 1e-4

        self.normal_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma)
        self.act_out = tf.reshape(self.normal_dist.sample(1), shape=[-1,1])
        self.act_out = tf.clip_by_value(self.act_out, self.action_low, self.action_high, name = "act_out")

        return self.mu,self.sigma,self.act_out

    def create_weights(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def create_bias(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def create_optimizer(self):
        # Using Adam to minimize the error between target and evaluation
        logprobs = self.normal_dist.log_prob(self.action)
        entropy = self.normal_dist.entropy()
        cost = tf.reduce_mean(-logprobs * self.G - 0.01*entropy)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name = "optimizer")

    def get_action(self, state):
        return self.act_out.eval(session = self.sess, feed_dict={self.state_input: [state]})[0]

    def get_action_set(self,state):
        return self.sess.run([self.act_out,self.mu,self.sigma],feed_dict={self.state_input:states})
 
    #def get_mu(self,state):
        #return self.mu.eval(session =self.sess,feed_dict={self.state_input:[state]})[0]
    #def get_sigma(self,state):
        #return self.sigma.eval(session = self.sess,feed_dict={self.state_input:[state]})[0]


    def save_model(self, step):
        # Helper function to save your model.
        with self.graph.as_default():
            saver = tf.train.Saver()
        self.sess.graph.add_to_collection("optimizer", self.optimizer)
        saver.save(self.sess, "./models/actor/actor", global_step = step)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(model_file + '.meta')
            saver.restore(self.sess, model_file)

        self.act_out = self.graph.get_tensor_by_name("act_out:0")
        self.state_input = self.graph.get_tensor_by_name("state_input:0")
        self.optimizer = self.graph.get_collection("optimizer")[0]

class Critic(object):
    def __init__(self, num_observation, lr, model = None):
        # define the network for the critic
        self.num_observation = num_observation
        self.learning_rate = lr

        if model != None:
            self.load_model(model)
        else:
<<<<<<< HEAD
            self.q_values =self.create_mlp()
            self.create_optimizer()
            self.sess.run(tf.global_variables_initializer())
=======
            self.graph = tf.Graph()
            self.sess = tf.Session(graph = self.graph)
            with self.graph.as_default():
                self.create_mlp()
                self.create_optimizer()
                self.sess.run(tf.global_variables_initializer())
>>>>>>> save and load a2c

    def train(self, states, R):
        self.optimizer.run(session=self.sess, feed_dict={self.state_input: states, self.target_q_value: R})

    def create_mlp(self):
        # Craete multilayer perceptron (one hidden layer with 20 units)
        self.hidden_units = 20

        self.w1 = self.create_weights([self.num_observation, self.hidden_units])
        self.b1 = self.create_bias([self.hidden_units])

        self.state_input = tf.placeholder(tf.float32, [None, self.num_observation], name = "state_input")

        h_layer = tf.nn.relu(tf.matmul(self.state_input, self.w1) + self.b1)

        self.w2 = self.create_weights([self.hidden_units, 1])
        self.b2 = self.create_bias([1])
        self.q_values = tf.add(tf.matmul(h_layer, self.w2), self.b2, name = "q_values")
        return self.q_values

    def create_weights(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def create_bias(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def create_optimizer(self):
        # Using Adam to minimize the error between target and evaluation
        self.target_q_value = tf.placeholder(tf.float32, [None, 1], name = "target_q_value")
        cost = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, self.q_values)))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name = "optimizer")

    def get_critics(self, states):
        return self.q_values.eval(session = self.sess, feed_dict={self.state_input: states})

    def save_model(self, step):
        # Helper function to save your model.
        with self.graph.as_default():
            saver = tf.train.Saver()
        self.sess.graph.add_to_collection("optimizer", self.optimizer)
        saver.save(self.sess, "./models/critic/critic", global_step = step)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(model_file + '.meta')
            saver.restore(self.sess, model_file)

        self.q_values = self.graph.get_tensor_by_name("q_values:0")
        self.state_input = self.graph.get_tensor_by_name("state_input:0")
        self.target_q_value = self.graph.get_tensor_by_name("target_q_value:0")
        self.optimizer = self.graph.get_collection("optimizer")[0]

from tensorflow.core.framework import summary_pb2
def summary_var(log_dir, name, val, step):
    writer = tf.summary.FileWriterCache.get(log_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = name
    value.simple_value = float(val)
    writer.add_summary(summary_proto, step)
    writer.flush()

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    N_step = args.N_step
    render = args.render

    # Create the environment.
    env = gym.make(ENVIROMENT)
    
    a2c = A2C(env, model_config_path, lr, critic_lr, num_episodes, N_step, render)#, model_step = 2500)

    a2c.train()
    # print(a2c.test(0, True))

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)