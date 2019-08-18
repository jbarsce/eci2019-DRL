import gfootball.env as football_env
from gfootball.env import football_action_set, scenario_builder
import torch
from dqn.dqn import Agent

env = football_env.create_environment(
    env_name='academy_empty_goal_close',
    stacked=False,                           # solo estado, no pixeles
    representation='simple115',              # solo estado, no pixeles
    with_checkpoints=True,                   # solo estado, no pixeles
    render=False                              # mostrar graficamente
)

football_action_set.action_set_dict['default']

agent = Agent(state_size=115, action_size=21, seed=0)

state = env.reset()
for j in range(200):
    action = agent.act(state)
    # if render: env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.close()

for i in range(1, 500):
    env.reset()
    acc_reward = 0

    done = False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        acc_reward += reward

    print("Recomensa episodio {:d}: {:.2f}".format(i, acc_reward))

env.close()

acc_reward = 0
# test code
for i in range(1, 50):
    env.reset()

    done=False
    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        acc_reward += reward

print("Recompensa promedio de los últimos 50 episodios {:.2f}".format(acc_reward/50))