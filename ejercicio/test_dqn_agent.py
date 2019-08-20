import gfootball.env as football_env
from gfootball.env import football_action_set, scenario_builder
import matplotlib.pyplot as plt
import numpy as np
from dqn.dqn import Agent
import logging
import torch

logging.disable(logging.WARNING)
render=False

env = football_env.create_environment(
    env_name='academy_empty_goal_close',
    stacked=False,                           # solo estado, no pixeles
    representation='simple115',              # solo estado, no pixeles
    with_checkpoints=True,                   # solo estado, no pixeles
    render=render                              # mostrar graficamente
)

football_action_set.action_set_dict['default']

agent = Agent(state_size=115, action_size=21, seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint_dqn.pth'))
reward_per_episode = []

# observamos el agente ya entrenado por 50 episodios
for i in range(50):
    state = env.reset()
    acc_reward = 0

    done = False
    while not done:
        action = agent.act(state)
        if render: env.render()
        state, reward, done, _ = env.step(action)
        acc_reward += reward

    reward_per_episode.append(acc_reward)

env.close()

print('Recompensa promedio: {}'.format(np.mean(reward_per_episode)))

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(reward_per_episode)), reward_per_episode)
plt.ylabel('Puntuacion')
plt.xlabel('Episodio #')
plt.show()