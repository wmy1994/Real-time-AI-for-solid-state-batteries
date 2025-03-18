import os.path

import matplotlib.pyplot as plt
from Env_SingleBat import sinBATenv

from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn import MlpPolicy, CnnPolicy
import wandb
import argparse
from datetime import datetime
from Parameters import General_param
Gen = General_param

if __name__ == '__main__':

    model_name = 'RL_checkpoint__10000_steps-22_August_2024_01h_26m.zip'
    checkpoint_path = os.path.join("./RL_model", model_name)
    test_step_num = 1000

    parser = argparse.ArgumentParser()
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-Re_time', type=str, default=None, help='time for data saving files')
    args = parser.parse_args()

    if args.resume:
        Gen.TIME_NOW = args.Re_time
        try:
            monitor_txt = 'post_process\\monitor_save-' + args.Re_time + '/cycling data.txt'
            reset_txt = 'post_process\\monitor_reset_save-' + args.Re_time + '/reset-cycling data.txt'
            with open(monitor_txt, "r") as file:
                done_steps = int(file.readlines()[-1].split("\t")[0])
            with open(reset_txt, "r") as file:
                done_episodes = int(file.readlines()[-1].split("\t")[0])

            Gen.current_steps = done_steps + 1
            Gen.current_episodes = done_episodes + 1
        except FileNotFoundError:
            print("No monitor_txt file, please check\n")
    else:
        DATE_FORMAT = Gen.DATE_FORMAT
        # time when we run the script
        Gen.TIME_NOW = datetime.now().strftime(DATE_FORMAT)

    env = sinBATenv()
    try:
        print('=== Try to Load:', checkpoint_path, '===')

        model = DQN.load(checkpoint_path, env=env)
        print('successfully loaded the model...', model_name)
    except FileNotFoundError:
        print("Not found model, please check\n")

    '''test part'''
    obs = env.reset()
    for _ in range(test_step_num):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

    print('battery test finished')

