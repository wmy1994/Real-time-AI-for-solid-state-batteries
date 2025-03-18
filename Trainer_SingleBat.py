import os
import matplotlib.pyplot as plt
from Env_SingleBat import sinBATenv

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.dqn import MlpPolicy, CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
import argparse
from datetime import datetime

from Parameters import General_param
Gen = General_param


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-Re_time', type=str, default=None, help='time for data saving files')
    parser.add_argument('-new_continue', action='store_true', default=False, help='continue training with a new battery')
    parser.add_argument('-continue_modelPath', type=str, default=None, help='the loaded model to continue training')
    args = parser.parse_args()

    if args.resume:
        Gen.TIME_NOW = args.Re_time
    else:
        DATE_FORMAT = Gen.DATE_FORMAT
        # time when we run the script
        Gen.TIME_NOW = datetime.now().strftime(DATE_FORMAT)

    total_timesteps = Gen.total_timesteps

    save_freq = 10
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path="./checkpoints/"+Gen.TIME_NOW,
                                             name_prefix="RL_checkpoint_")

    if not args.resume:
        if not args.new_continue:
            env = sinBATenv()
            model = DQN(CnnPolicy, env, verbose=1, buffer_size=100, batch_size=8)

        else:
            try:
                checkpoint_path = args.continue_modelPath
                print('=== Try to Load:', checkpoint_path, '===')

                env = sinBATenv()
                model = DQN.load(checkpoint_path, env=env)
                print('successfully loaded the model...')
                print('model current timesteps:', model.num_timesteps, '\n')

            except FileNotFoundError:
                print("No continued model, please restart training\n")
    else:
        try:
            monitor_txt = 'post_process\\monitor_save-' + args.Re_time + '/cycling data.txt'
            reset_txt = 'post_process\\monitor_reset_save-' + args.Re_time + '/reset-cycling data.txt'
            with open(monitor_txt, "r") as file:
                done_steps = int(file.readlines()[-1].split("\t")[0])
            with open(reset_txt, "r") as file:
                done_episodes = int(file.readlines()[-1].split("\t")[0])

            total_timesteps = Gen.total_timesteps - done_steps
            print(f"\n=== Successful reading historical model step number = {done_steps} ===")

            check_root = "./checkpoints/"+Gen.TIME_NOW
            checkpoint_list = os.listdir(check_root)
            checkpoint_list = sorted(checkpoint_list, key=lambda x: int(x.split('_')[3]))

            checkpoint_path = os.path.join("./checkpoints/" + Gen.TIME_NOW, checkpoint_list[-1])

            print('=== Try to Load:', checkpoint_path,'===')

            Gen.current_steps = done_steps + 1
            Gen.current_episodes = done_episodes + 1
            env = sinBATenv()

            if os.path.exists(checkpoint_path):
                model = DQN.load(checkpoint_path, env=env)
                print('Successfully loaded the model...')
                print('model current timesteps:', model.num_timesteps, '\n')

        except FileNotFoundError:
            print("No monitor_txt file, please restart training\n")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, reset_num_timesteps=False)

    model.save("./RL_model/dqn_single_optim_"+Gen.TIME_NOW)

    print('RL training finished')
