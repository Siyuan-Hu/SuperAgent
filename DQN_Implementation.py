#!/usr/bin/env python
import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse

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



class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name, actor_mimic = False, model = None):
		# Define your network architecture here. It is also a good idea to define any training operations 
		# and optimizers here, initialize your variables, or alternately compile your model here.  
		self.is_actor_mimic = actor_mimic
		env = gym.make(environment_name)
		self.state_dim = list(env.observation_space.shape)

		# The shape of the origin state could have multiple dimensions.
		# We flat the state dimensions here
		self.flat_state_dim = 1
		for i in self.state_dim:
			self.flat_state_dim *= i

		self.action_dim = env.action_space.n
		self.learning_rate = 0.0001

		self.session = tf.InteractiveSession()

		if model != None:
			self.load_model(model)
		else:
			self.create_mlp()
			self.create_optimizer()

		env.close()

	def create_weights(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.1)
		return tf.Variable(initial)

	def create_bias(self, shape):
		initial = tf.constant(0.1, shape = shape)
		return tf.Variable(initial)

	def create_mlp(self):
		# Craete multilayer perceptron (one hidden layer with 20 units)
		self.hidden_units = 20

		self.w1 = self.create_weights([self.flat_state_dim, self.hidden_units])
		self.b1 = self.create_bias([self.hidden_units])

		self.state_input = tf.placeholder(tf.float32, [None] + self.state_dim, name = "state_input")

		flat_state = tf.reshape(self.state_input, [-1, self.flat_state_dim])

		h_layer = tf.nn.relu(tf.matmul(flat_state, self.w1) + self.b1)

		self.w2 = self.create_weights([self.hidden_units, self.action_dim])
		self.b2 = self.create_bias([self.action_dim])

		self.q_values = tf.add(tf.matmul(h_layer, self.w2), self.b2, name = "q_values")
		if self.is_actor_mimic:
			self.q_values = tf.nn.softmax(self.q_values, name = "q_values")

	def create_optimizer(self):
		# Using Adam to minimize the error between target and evaluation
		if self.is_actor_mimic:
			cost = self.actor_mimic_cost()
		else:
			cost = self.dqn_cost()

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost, name = "optimizer")

		self.session.run(tf.global_variables_initializer())

	def dqn_cost(self):
		self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name = "action_input")
		self.target_q_value = tf.placeholder(tf.float32, [None], name = "target_q_value")
		q_value_output = tf.reduce_sum(tf.multiply(self.q_values, self.action_input), 1)
		cost = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, q_value_output)))
		return cost

	def actor_mimic_cost(self):
		self.expert_q_values = tf.placeholder(tf.float32, [None, self.action_dim])
		cost = - tf.reduce_mean(tf.reduce_sum(tf.multiply(expert_q_values, tf.log(self.q_values))))
		return cost

	def update_dqn(self, state_batch, action_batch, target_batch):
		self.optimizer.run(feed_dict = {self.state_input : state_batch, 
			self.action_input : action_batch, self.target_q_value : target_batch})

	def update_actor_mimic_network(state_batch, expert_q_values_batch):
		self.optimizer.run(feed_dict = {self.state_input : state_batch, 
			self.expert_q_values : expert_q_values_batch})

	def get_boltzmann_distribution_over_q_values(self, state_batch):
		temperature = 1.0
		distribution = tf.exp(self.q_values / temperature)
		dist_sum = tf.reduce_sum(distribution, 1)
		return (distribution/dist_sum).eval(feed_dict = {self.state_input : state_batch})

	def get_q_values(self, state):
		return self.q_values.eval(feed_dict = {self.state_input : [state]})[0]


	def save_model(self, suffix, step):
		# Helper function to save your model.
		saver = tf.train.Saver()
		tf.add_to_collection("optimizer", self.optimizer)
		saver.save(self.session, suffix, global_step = step)

	def load_model(self, model_file):
		# Helper function to load an existing model.
		saver = tf.train.import_meta_graph(model_file + '.meta')
		saver.restore(self.session, model_file)

		graph = tf.get_default_graph()
		self.q_values = graph.get_tensor_by_name("q_values:0")
		self.state_input = graph.get_tensor_by_name("state_input:0")
		self.action_input = graph.get_tensor_by_name("action_input:0")
		self.target_q_value = graph.get_tensor_by_name("target_q_value:0")
		self.optimizer = tf.get_collection("optimizer")[0]

	def get_weight(self):
		return self.w2, self.b2

	def set_weight(self, w, b):
		self.w2 = w
		self.b2 = b

class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		pass

	def sample_batch(self, batch_size=32):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		pass

	def append(self, transition):
		# Appends transition to the memory. 	
		pass

class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 

		pass 

	def epsilon_greedy_policy(self, q_values):
		# Creating epsilon greedy probabilities to sample from.             
		pass

	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		pass 

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.
		pass

	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 

	def burn_in_memory():
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		pass

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	gpu_ops = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_ops)
	sess = tf.Session(config=config)

	# Setting this as the default tensorflow session. 
	keras.backend.tensorflow_backend.set_session(sess)

	# You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
	main(sys.argv)

