import gfootball.env as football_env
from gfootball.env import football_action_set, scenario_builder

env = football_env.create_environment(
    env_name='academy_empty_goal_close',
    stacked=False,                           # solo estado, no pixeles
    representation='simple115',              # solo estado, no pixeles
    with_checkpoints=True,                   # solo estado, no pixeles
    render=True                              # mostrar graficamente
)

football_action_set.action_set_dict['default']

for i in range(1, 10):
    env.reset()
    acc_reward = 0

    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        acc_reward += reward

        if done:
            break

    print("Recomensa episodio {:d}: {:.2f}".format(i, acc_reward))

env.close()