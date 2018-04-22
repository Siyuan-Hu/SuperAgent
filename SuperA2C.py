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

class StudentNetwork(object):
	"""docstring for StudentNetwork"""
	def __init__(self, num_observation, actor_lr, critic_lr, action_low, action_high):
		self.actor = Actor(num_observation=num_observation,
						   lr=actor_lr,
						   action_low=action_low,
						   action_high=action_high)
		self.target_mu = tf.placeholder(tf.float32, [None, 1])
		self.target_sigma = tf.placeholder(tf.float32, [None, 1])
		actor_loss = tf.reduce_mean(tf.square(tf.subtract(self.target_mu, self.actor.mu)) + tf.square(tf.subtract(self.target_sigma, self.actor.sigma)) )
		self.actor_optimizer = tf.train.AdamOptimizer(actor_lr).minimize(actor_loss, name="actor_optimizer")
		self.actor.sess.run(tf.global_variables_initializer())


		self.critic = Critic(num_observation=num_observation,
							 lr=critic_lr)
		self.target_v = tf.placeholder(tf.float32, [None, 1])
		critic_loss = tf.reduce_mean(tf.square(tf.subtract(self.target_v, self.critic.q_values)))
		self.critic_optimizer = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss, name="critic_optimizer")
		self.critic.sess.run(tf.global_variables_initializer())

	def train_actor(self, states, mu, sigma):
		self.actor_optimizer.run(session=self.actor.sess, feed_dict={self.actor.state_input: states,
																	 self.target_mu: mu,
																	 self.target_sigma: sigma})

	def train_critic(self, states, v):
		self.critic_optimizer.run(session=self.critic.sess, feed_dict={self.critic.state_input: states,
																	   self.target_v: v})

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

		self.action_high = self.env.action_space.high
		self.action_low = self.env.action_space.low
		self.actor_net = Actor(self.num_observation,self.actor_learning_rate,self.action_low,self.action_high)
		self.critic_net = Critic(self.num_observation,self.critic_learning_rate)
		# use monitor to generate video

		if (open_monitor):
			video_save_path = self.save_path + "/video/"
			self.env = gym.wrappers.Monitor(self.env,
											video_save_path,
											resume=True)

		# parameter for the network
		# self.resume = resume # whether use the pre-trained model

		self.network_name = network_name


		self.teach_burn_in_memory()
		# flag to keep the status for enviroment in the teach mode

		self.teach_current_state = self.initialize_env(self.env)




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

	def teach(self,env):
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

		for tmp_state,tmp_mu,tmp_sigma,tmp_q_values in batch:
			batch_state_lst.append(tmp_state)
			batch_mu_lst.append(tmp_mu)
			batch_sigma_lst.append(tmp_sigma)
			batch_q_values_lst.append(tmp_q_values)
			#batch_action_lst.append(tmp_action)

		return batch_state_lst, batch_mu_lst,batch_sigma_lst,batch_q_values_lst


	def teach_burn_in_memory(self):
		
		env = gym.make(self.environment_name)
		done = False
		current_state = self.initialize_env(env)

		for i in range(self.burn_in):
			q_values = self.get_target_q(current_state)
			action = self.get_target_action(current_state)
			mu = self.get_target_mu(current_state)
			sigma = self.get_target_sigma(current_state)
			next_state, _,done, _ = self.get_next_state(action,
														 env)
			self.replay_memory.append((current_state,
										mu,
										sigma,
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
