import gfootball.env as football_env
from gfootball.env import football_action_set, scenario_builder
import matplotlib.pyplot as plt
import numpy as np
from ddpg.ddpg_agent import Agent
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

# hyper-parameters
gamma = 0.99

max_episodes = 1000

agent = Agent(state_size=115, action_size=21, random_seed=0)
reward_per_episode = []

for episode_i in range(1, max_episodes):
    state = env.reset()
    acc_reward = 0

    done = False
    while not done:
        actions = agent.act(state)

        actions_softmax = np.exp(actions)/np.sum(np.exp(actions))

        action = np.random.choice(np.arange(21), p=actions_softmax)
        next_state, reward, done, info = env.step(action)
        acc_reward += reward

        # almacenar <St, At, Rt+1, St+1>
        agent.memory.add(state, actions, reward, next_state, done)

        # train & update
        agent.step(state, actions, reward, next_state, done)

        # avanzar estado
        state = next_state

    reward_per_episode.append(acc_reward)

    if episode_i % 50 == 0 or episode_i == (max_episodes-1):
        print("Recomensa últimos 50 episodios de episodio {:d}: {:.2f}".format(episode_i, np.mean(reward_per_episode[-50:])))

torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

env.close()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(reward_per_episode)), reward_per_episode)
plt.ylabel('Puntuacion')
plt.xlabel('Episodio #')
plt.show()