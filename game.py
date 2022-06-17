
from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import greedy_agent
from agents.ppo.ppo_v1.ppo_v1 import Agent as ppo1
from agents.ppo.ppo_v2.ppo_v2 import Agent as ppo2
# from agents.ppo.ppo_v4.ppo_v4 import Agent as ppo4
from submission.ppo4.ppo4_subm import agent as ppo4
env = make("hungry_geese", debug=True)
env.reset()
env.run([ppo4, ppo1(), ppo2(), greedy_agent])
