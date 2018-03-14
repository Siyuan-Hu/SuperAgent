#!/usr/bin/env python
import keras, tensorflow as tf, numpy as npy, gym, sys, copy, argparse

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        pass

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        pass

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        pass

class Replay_Memory():

    def __init__(self,
                 batch_size=16,
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

class DQN_Agent():

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
                 save_path,
                 logger,
                 render=False,
                 episodes=10000,
                 epsilon_begin=0.5,
                 epsilon_end=0.05,
                 gamma=0.99,
                 learning_rate=0.00001,
                 train_model=0,
                 teach_model=0,
                 resume=0,
                 batch_size=16,
                 memory_size=50000,
                 burn_in=10000,
                 open_monitor=0,
                 frequency_update=1000,
                 frequency_sychronize=10):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 

        # parameter for the gym enviroment
        self.environment_name = environment_name
        self.logger = logger
        self.render = render
        self.episodes = episodes
        self.epsilon_begin = epsilon_begin
        self.epsilon_end = epsilon_end
        self.gamma = gamma

        # parameter from the enviroment
        self.env = gym.make(self.environment_name)
        self.num_actions = self.env.action_space.n
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
        self.burn_in_memory()

        # use monitor to generate video
        if (open_monitor):
            video_save_path = self.save_path + "/video/"
            self.env = gym.wrappers.Monitor(self.env,
                                            video_save_path,
                                            resume=True)

        # parameter for the network
        # self.resume = resume # whether use the pre-trained model
        self.learning_rate = learning_rate
        self.network_name = network_name

        if train_model == teach_model:
            raise Exception("Wrong agent model, agent can only do one thing between train and teach model")

        self.train_model = train_model # use this agent to train the teacher
        self.teach_model = teach_model # use this agent as a teacher to teach the student

        # initilize the network
        self.q_network = QNetwork(self.env,
                                  self.network_name,
                                  self.learning_rate)

        ## TODO
        if (resume):
            # self.q_network.load_model(pre-train_model_path)

        # if this agent is just to teach,
        # then there is no need of target_network
        if (train_model):
            self.target_network = QNetwork(self.env,
                                           self.network_name,
                                           self.learning_rate)
            ## TODO
            # sychronize the q network and target network
            self.target_network.sychronize_all_weights(self.q_network.get_all_weights())


    def get_next_state(self, action, env):
        # given the action, return next state, reward, done and info

        next_state, reward, done, info = env.step(action)

        return np.array([next_state]), reward, done, info

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.

        if (epsilon < np.random.rand()):
            greedy_policy = self.greedy_policy(q_values)
        else:
            greedy_policy = self.env.action_space.sample()

        return greedy_policy

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        
        return np.argmax(q_values)

    def initialize_env(self, env):
        # reset the env

        state = env.reset()

        return np.array([state])

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.


        # reward = 0
        episode_count = 0
        update_count = 0
        # iteration_count = 0
        test_reward = []
        # current_state = np.zeros(self.num_observation)
        # next_state = np.zeros(self.num_observation)
        # target = np.zeros(self.num_actions)
        epsilon_array = np.linspace(self.epsilon_begin,
                                    self.epsilon_end,
                                    num=self.episodes)
        for epsilon in epsilon_array:
            
            current_state = self.initialize_env(self.env)
            done = False
            reward_sum = 0
            while not done:
                
                if episode_count % 20 == 0 and self.render:
                    self.env.render()
                ## TODO 
                # need network module predict function
                q_values = self.q_network.predict(current_state)
                action = self.epsilon_greedy_policy(q_values,
                                                    epsilon)
                next_state, reward, done, info = self.get_next_state(action,
                                                                     self.env)
                reward_sum += reward

                # append to memory
                self.replay_memory.append((current_state,
                                           action,
                                           reward,
                                           next_state,
                                           done))

                # sample from the memory
                batch = self.replay_memory.sample()
                batch_state_lst = []
                batch_q_target_lst = []
                for tmp_state, tmp_action, tmp_reward, tmp_next_state, tmp_done in batch:
                    ## TODO
                    ## this can be done in batch maybe
                    ## this maybe need to check again
                    q_target = self.q_network.predict(tmp_state)
                    if (tmp_done):
                        q_target[0][tmp_action] = tmp_reward
                    else:
                        q_target[0][tmp_action] = tmp_reward + self.gamma * np.amax(self.target_network.predict(tmp_next_state))
                    batch_state_lst.append(tmp_state)
                    batch_q_target_lst.append(q_target[0])

                batch_state = np.vstack(batch_state_lst)
                batch_q_target = np.array(batch_q_target_lst)

                ## TODO
                self.q_network.fit(batch_state,
                                   batch_q_target)

                # save and test q network
                # frequency_update = 10000
                if update_count % self.frequency_update == 0:
                    self.test(update_count)
                    self.q_network.save_model_weights(self.save_path, update_count)

                # sychronize the q_network with the target_network
                # frequency_sychronize = 100
                if update_count % self.frequency_sychronize == 0:
                    ## TODO
                    self.target_network.sychronize_all_weights(self.q_network.get_all_weights())

                # prepare for next
                current_state = next_state
                if done:
                    self.logger.info('Episode[%d], Epsilon=%f, Train-Reward Sum=%f',
                                episode_count,
                                epsilon,
                                reward_sum)

                update_count += 1
            episode_count += 1


    def test(self, update_count model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 

        reward_sum_lst = []
        env = gym.make(self.environment_name)
        num_test = 10
        epsilon = 0.05
        for _ in range(num_test):
            done = False
            current_state = self.initialize_env(env)
            reward_sum = 0
            while not done:
                q_values = self.q_network.predict(current_state)
                action = self.epsilon_greedy_policy(q_values,
                                                    epsilon)
                next_state, reward, done, info = self.get_next_state(action,
                                                                     env)
                reward_sum += reward
                current_state = next_state
                if done:
                    reward_sum_lst.append(reward_sum)

        self.logger.info('Update[%d], Test-Reward: Mean=%f, STD=%f',
                    update_count,
                    np.mean(reward_sum_lst),
                    np.std(reward_sum_lst))


    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions. 

        ## TODO
        # to check use which env and Monitor
        env = gym.make(self.environment_name)
        done = False
        current_state = self.initialize_env(env)
        for i in range(self.burn_in):
            action = env.action_space.sample()
            next_state, reward, done, info = self.get_next_state(action, env)
            self.replay_memory.append((current_state,
                                       action,
                                       reward,
                                       next_state,
                                       done))
            if done:
                current_state = self.initialize_env(self.env)
            else:
                current_state = next_state

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
    args = parse_arguments()
    p_attrs = vars(args)
    for k, v in sorted(p_attrs.items(), key=lambda x: x[0]):
        logging.info("%s : %s", k, v)


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

