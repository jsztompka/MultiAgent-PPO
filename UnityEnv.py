import numpy as np
from unityagents import UnityEnvironment

"""UnityEnv is a wrapper around UnityEnvironment 
    The main purpose for this Env is to establish a common interface which most environments expose
    """

class UnityEnv:
    def __init__(self,
                 env_path,
                 train_mode = True
                 ):

        self.brain = None
        self.brain_name = None

        self.train_mode = train_mode
        self.env = self.create_unity_env(env_path)

        #env details
        self.action_space = self.brain.vector_action_space_size
        self.observation_space = self.brain.vector_observation_space_size

        print(f'Action space {self.action_space}')
        print(f'State space {self.observation_space}')



        #backwards compatibility
        self.action_dim = self.action_space
        #self.observation_space = self.env.observation_space
        self.state_dim = int(np.prod(self.observation_space))



    def extract_env_details(self, env_info):
        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished

        return next_state, reward, done

    def create_unity_env(self, env_path):
        env = UnityEnvironment(file_name=env_path)

        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

        return env

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return self.extract_env_details(env_info)[0]

    def step(self, actions):

        actions = np.clip(actions, -1, 1)
        # torch.clamp(actions, min=-1, max=1)

        self.env.step(actions)[self.brain_name]


        env_info = self.env.step(actions)[self.brain_name]
        next_states, rewards, dones = self.extract_env_details(env_info)

        return next_states, rewards, np.array(dones)

        # return next_state, reward, np.array([done])