from torchsummary import summary
from model import PPONetwork
from PPO_agent import PPOAgent_Unity
import config

if __name__ == '__main__':
    agent = PPOAgent_Unity(config.Config())

    #run 100 episodes
    for i in range(100):
        print(f"Agent step={i}")
        agent.step()

