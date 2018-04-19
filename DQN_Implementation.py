#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
from A2C_Continuous import Actor,Critic
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

	def __init__(self, environment_name, actor_mimic = False, model = None, learning_rate = 0.0001):
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
		self.learning_rate = learning_rate

		self.session = tf.InteractiveSession(graph = tf.Graph())

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
			self.cost = self.actor_mimic_cost()
		else:
			self.cost = self.dqn_cost()

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost, name = "optimizer")
		tf.add_to_collection("optimizer", self.optimizer)
		self.session.run(tf.global_variables_initializer())

	def dqn_cost(self):
		self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name = "action_input")
		self.target_q_value = tf.placeholder(tf.float32, [None], name = "target_q_value")
		q_value_output = tf.reduce_sum(tf.multiply(self.q_values, self.action_input), 1)
		cost = tf.reduce_mean(tf.square(tf.subtract(self.target_q_value, q_value_output)))
		return cost

	def actor_mimic_cost(self):
		self.expert_q_values = tf.placeholder(tf.float32, [None, self.action_dim])
		cost = - tf.reduce_mean(tf.reduce_sum(tf.multiply(self.expert_q_values, tf.log(self.q_values)), 1))
		return cost

	def get_actor_mimic_cost(self, state_batch, expert_q_values_batch):
		return self.cost.eval(session = self.session, feed_dict = {self.state_input : state_batch, 
			self.expert_q_values : expert_q_values_batch})

	def update_dqn(self, state_batch, action_batch, target_batch):
		self.optimizer.run(session = self.session, feed_dict = {self.state_input : state_batch, 
			self.action_input : action_batch, self.target_q_value : target_batch})

	def update_actor_mimic_network(self, state_batch, expert_q_values_batch):
		self.optimizer.run(session = self.session, feed_dict = {self.state_input : state_batch, 
			self.expert_q_values : expert_q_values_batch})

	def get_boltzmann_distribution_over_q_values(self, state):
		q_values = self.get_q_values(state)

		temperature = 1.0
		distribution = np.exp(q_values / temperature)
		dist_sum = np.sum(distribution)
		return distribution / dist_sum

	def get_q_values(self, state):
		return self.q_values.eval(session = self.session, feed_dict = {self.state_input : [state]})[0]


	def save_model(self, suffix, step):
		# Helper function to save your model.
		saver = tf.train.Saver()
		# tf.add_to_collection("optimizer", self.optimizer)
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

	def __init__(self,
				 batch_size=4,
				 memory_size=50000,
				 burn_in=10000):
		import collections
		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
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

class Agent():

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
				 network_name,
				 logger,
				 save_path='tmp',
				 render=False,
				 episodes=1000000,
				 gamma=0.99,
				 actor_learning_rate=0.00001,
				 critic_learning_rate = 0.0005,
				 noise = 0.01,
				 train_model=1,
				 teach_model=0,
				 model=None,
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
		self.episodes = episodes
		self.gamma = gamma
		self.noise = noise

		# parameter from the enviroment
		self.env = gym.make(self.environment_name)
		self.num_actions = self.env.action_space.shape[0]
		self.num_observation = self.env.observation_space.shape[0]

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
		self.actor_net = Actor(self.num_observation,self.actor_learning_rate,self.env.action_space.low,self.env.action_space.high)
		self.critic_net = Critic(self.num_observation,self.critic_learning_rate)
		# use monitor to generate video
		if (open_monitor):
			video_save_path = self.save_path + "/video/"
			self.env = gym.wrappers.Monitor(self.env,
											video_save_path,
											resume=True)

		# parameter for the network
		# self.resume = resume # whether use the pre-trained model
		self.actor_learning_rate = actor_learning_rate
		self.critic_learning_rate = critic_learning_rate
		self.network_name = network_name

		if model == None:
			self.q_network = QNetwork(self.environment_name)
		else:
			self.q_network = QNetwork(self.environment_name, model=model)


		if train_model == teach_model:
			raise Exception("Wrong agent model, agent can only do one thing between train and teach model")
		elif (train_model):
			self.burn_in_memory()
		elif (teach_model):
			self.teach_burn_in_memory()
			# flag to keep the status for enviroment in the teach mode
			self.teach_done = False
			self.teach_current_state = self.initialize_env(self.env)

			
		self.train_model = train_model # use this agent to train the teacher
		self.teach_model = teach_model # use this agent as a teacher to teach the student


		# # if this agent is just to teach,
		# # then there is no need of target_network
		# if (train_model):
		#     self.target_network = QNetwork(self.env,
		#                                    self.network_name,
		#                                    self.learning_rate)
		#     ## TODO
		#     # sychronize the q network and target network
		#     self.target_network.sychronize_all_weights(self.q_network.get_all_weights())


	def get_next_state(self, action, env):
		# given the action, return next state, reward, done and info

		next_state, reward, done, info = env.step(action)

		# return np.array([next_state]), reward, done, info
		return next_state, reward, done, info

	def get_actions(self,state):
		# need to change this line
		return self.q_network.get_action(state)

	def get_target_action(self,state):
		return self.actor_net.get_action(state)


	def get_target_q(self,state):
		return self.critic_net.get_critics(state)

	def get_target_mu(self,state):
		return self.actor_net.get_mu(state)

	def get_target_sigma(self,state):
		return self.actor_net.get_sigma(state)

	def initialize_env(self, env):
		# reset the env

		return env.reset()

	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# If you are using a replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		episode_count = 0
		update_count = 0
		test_reward = []


		current_state = self.initialize_env(self.env)
		done = False
		reward_sum = 0

		while not done:
			
			if episode_count % 20 == 0 and self.render:
				self.env.render()

			## TODO 
			# need network module get_q_value function
			# need to modify line 329 330 
			q_values = self.q_network.get_q_values(current_state)
			action = self.get_actions(current_state)
			mu = self.get_target_mu(current_state)
			sigma = self.get_target_sigma(current_state)
			#action = self.epsilon_greedy_policy(q_values,
			#									epsilon)
			next_state, _, _, _ = self.get_next_state(action,self.env)


			# append to memory
			self.replay_memory.append((current_state,
									   action,
									   q_values
									   ))	

			# sample from the memory
			batch = self.replay_memory.sample()
			batch_state_lst = []
			batch_critic_target_lst = []
			batch_action_lst = []
			for tmp_state, tmp_action, tmp_critic in batch:
				## TODO
				## this can be done in batch maybe
				## this maybe need to check again
				action_target = self.get_target_action(env,current_state)
				#Actor(self.num_observation,self.actor_learning_rate,env.action_space.low,env.action_space.high).get_action(tmp_state)
				q_target = self.get_target_q(env,current_state)
				#Critic(self.num_observation,self.critic_learning_rate).get_critics(tmp_state)
                
				batch_state_lst.append(tmp_state)
				batch_q_target_lst.append(q_target)
				batch_action_lst.append(action_target)

			## TODO
			self.q_network.update_dqn(batch_state_lst,
									  batch_action_lst,
									  batch_q_target_lst)

			current_state = next_state

			update_count += 1
		env.close()
		if episode_count % 200 == 0:
			self.test()
		episode_count += 1




	def test(self):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 

		episode_num = 1
		total_reward = 0

		env = gym.make(self.environment_name)


		for i in range(episode_num):
			state = env.reset()
			done = False
			while not done:
				q_values = self.q_network.get_q_values(state)

				env.render()
				action = self.get_actions(state)

				state, reward, done, info = env.step(action)

				total_reward += reward

		ave_reward = total_reward / episode_num
		print ('Evaluation Average Reward:', ave_reward)    
		env.close()
		return ave_reward


	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions. 

		## TODO
		# to check use which env and Monitor
		env = gym.make(self.environment_name)
		done = False
		current_state = self.initialize_env(env)
		for i in range(self.burn_in):
			#a_high = env.action_space.high
			#a_low = env.action_space.low
			#action = self.get_target_action(current_state)
			#mu = self.get_target_mu(current_state)
			#sigma = self.get_target_sigma(current_state)
			#action = np.clip(np.random.normal(loc = mu,scale=sigma),env.action_space.low,env.action_space.high)

			#+np.random.randn(num_actions)*self.noise
			#Actor(self.num_observation,self.actor_learning_rate,env.action_space.low,env.action_space.high).get_action(tmp_state)
			action = env.action_space.sample()
			next_state, reward, done, info = self.get_next_state(action, env)
			critic_value = self.get_target_q(current_state)
			self.replay_memory.append((current_state,
									   action,
									   critic_value))
			if done:
				current_state = self.initialize_env(env)
			else:
				current_state = next_state
		env.close()


	def teach(self,env):
		# every time call this func, 
		# (1), it will append a new episode into the replay memory with
		#      every step's current_state and q_values
		# (2), return the sample batch of current_state and q_values


		current_state = self.teach_current_state

		q_values = self.q_network.get_boltzmann_distribution_over_q_values(current_state)
		action = self.get_actions(current_state)
		#mu = self.get_target_mu(current_state)
		#sigma = self.get_target_sigma(current_state)
		#action = np.clip(np.random.normal(loc = mu,scale=sigma),env.action_space.low,env.action_space.high)

		next_state, _, done, _ = self.get_next_state(action,
													 self.env)
		self.replay_memory.append((current_state,
									action,
								   q_values))

		if done:
			self.teach_current_state = self.initialize_env(self.env)
		else:
			self.teach_current_state = next_state


		batch = self.replay_memory.sample()

		batch_state_lst = []
		batch_action_lst = []
		batch_q_values_lst = []
		for tmp_state,tmp_action ,tmp_q_values in batch:
			batch_state_lst.append(tmp_state)
			batch_q_values_lst.append(tmp_q_values)
			batch_action_lst.append(tmp_action)
		return batch_state_lst, batch_action_lst,batch_q_values_lst


	def teach_burn_in_memory(self):
		
		env = gym.make(self.environment_name)
		done = False
		current_state = self.initialize_env(env)

		for i in range(self.burn_in):
			q_values = self.q_network.get_boltzmann_distribution_over_q_values(current_state)
			action = self.get_actions(current_state)

			next_state, _, done, _ = self.get_next_state(action,
														 env)
			self.replay_memory.append((current_state,
										action,
									   q_values))
			if done:
				current_state = self.initialize_env(env)
			else:
				current_state = next_state
		env.close()

def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()

def main(args):
	import logging
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	# # args = parse_arguments()
	# # p_attrs = vars(args)
	# # for k, v in sorted(p_attrs.items(), key=lambda x: x[0]):
	# #     logging.info("%s : %s", k, v)


	# # environment_name = args.env

	# # # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	# # gpu_ops = tf.GPUOptions(allow_growth=True)
	# # config = tf.ConfigProto(gpu_options=gpu_ops)
	# # sess = tf.Session(config=config)

	# # # Setting this as the default tensorflow session. 
	# # keras.backend.tensorflow_backend.set_session(sess)

	# # # You want to create an instance of the DQN_Agent class here, and then train / test it. 
	# # environment_name = "CartPole-v0"
	# # agent = DQN_Agent(environment_name,
	# #                   network_name='mlp',
	# #                   logger=logger)
	# # agent.train()

	num_update = 10000000
	environment_name_lst = ["MountainCar-v0", "Acrobot-v1"]
	model_path_lst = ["./expert/mountaincar/MountainCar-v0-243",
					  "./expert/acrobot/Acrobot-0"]
	teacher_agent_lst = []
	student_network_lst = []
	num_env = len(environment_name_lst)
	frequency_report_loss = 100
	# initilze the teacher agent and student network
	for idx in range(num_env):
		teacher_agent_lst.append(DQN_Agent(environment_name=environment_name_lst[idx],
										   network_name='mlp',
										   logger=logger,
										   model=model_path_lst[idx],
										   train_model=0,
										   teach_model=1,
										   burn_in=10000,
										   batch_size=10))
		student_network_lst.append(QNetwork(environment_name=environment_name_lst[idx],
											actor_mimic=True,
											learning_rate = 0.01))

		
	loss = 0
	for idx_update in range(num_update):
		for idx in range(num_env):
			teacher_agent = teacher_agent_lst[idx]
			student_network = student_network_lst[idx]
			## TODO
			# need future work(student provides state, teacher provides best q value) 
			batch_state_lst, batch_q_values_lst = teacher_agent.teach()

			## TODO
			# whether update one network in the list will update
			# the network in the list
			student_network.update_actor_mimic_network(batch_state_lst,
													   batch_q_values_lst)
			loss += student_network.get_actor_mimic_cost(batch_state_lst, batch_q_values_lst)

			next_student_network_idx = (idx + 1) % num_env
			next_student_network = student_network_lst[next_student_network_idx]

			w, b = student_network.get_weight()
			next_student_network.set_weight(w, b)


			if ((idx_update % frequency_report_loss) == 0):
				print(loss / frequency_report_loss)
				# student_network.save_model("./" + environment_name_lst[idx], idx_update)
				student_agent = DQN_Agent(environment_name_lst[idx],
										   network_name='mlp',
										   logger=logger,
										   # model= "./" + environment_name_lst[idx]+"-"+str(idx_update),
										   train_model=0,
										   teach_model=1,
										   burn_in=0)
				student_agent.q_network = student_network
				student_agent.test()
				loss = 0



	# environment_name = "Acrobot-v1"
	# agent = DQN_Agent(environment_name,
	#                   network_name='mlp',
	#                   model="./expert/acrobot/Acrobot-0", 
	#                   logger=logger)
	# agent.test()

if __name__ == '__main__':
	main(sys.argv)

