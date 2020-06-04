#! /usr/bin/env python3
"""
Reload and reuse the trained PPO/TRPO policies

Adapted code, original code from Sanjay Thakur (sanjaykthakur.com)
"""

import tensorflow as tf
import _pickle as pickle
import gym, os, argparse
from gym import wrappers
import numpy as np
from datetime import datetime
from tqdm import tqdm
from skimage import transform, color
import torch
import random
from pathlib import Path
import h5py


def get_scale_and_offset(env_name: str):
    """
    Returns the values that were used to normalize to derive the saved PPO/TRPO models.

    Parameters:
        env_name (string):  Name of the environment whose normalization values are to be fetched

    Returns: The scale and the offset values
    """
    file_name = 'saved_models/' + env_name + '/scale_and_offset.pkl'
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['SCALE'], data['OFFSET']


def apply_learnt_policy_for_episode(sess, env, output_action_node, scaled_observation_node, offset, scale, pane=None):
    if pane == 'left':
        pane_check = False
        while not pane_check:
            observation = env.reset()
            # We ensure that the target (the ball) is generated in the left pane
            pane_check = env.env.env.get_body_com('target')[0] <= 0
    elif pane == 'right':
        pane_check = False
        while not pane_check:
            observation = env.reset()
            # We ensure that the target (the ball) is generated in the right pane
            pane_check = env.env.env.get_body_com('target')[0] >= 0
    else:
        observation = env.reset()
    
    done = False
    total_reward = 0.
    time_step = 0.

    observes, actions, rewards, unscaled_obs = [], [], [], []
    l_px_obs = []
    while not done:
        l_px_obs.append(get_pixel_obs(env))
        observation = observation.astype(np.float32).reshape((1, -1))
        observation = np.append(observation, [[time_step]], axis=1)  # add time step feature
        unscaled_obs.append(observation)
        obs = (observation - offset) * scale
        observes.append(obs)
        action = sess.run(output_action_node, feed_dict={scaled_observation_node: obs})
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        time_step += 1e-3

    return (np.stack(l_px_obs, axis=0), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def apply_learnt_policy(episodes, sess, env, output_action_node, scaled_observation_node, offset, scale, pane=None):
    total_steps = 0
    trajectories = []
    for e in tqdm(range(episodes)):
        px_observes, actions, rewards, unscaled_obs = apply_learnt_policy_for_episode(sess, env, output_action_node,
                                                                                   scaled_observation_node, offset,
                                                                                   scale, pane=pane)
        total_steps += px_observes.shape[0]
        trajectory = {'px_obs': px_observes,
                      'actions': actions,
                      'rewards': rewards}
        trajectories.append(trajectory)

    return trajectories

def get_pixel_obs(env):
    x = env.render(mode='rgb_array')
    # cant convert now else it takes too much space.
    resized = transform.resize(x, (84, 84, 3))
    return np.uint8(255*resized)
    #return x

def create_trajs(env_name: str, episodes, pane=None, color=None):
    """
    Loads the trained models, values for normalization, and runs them on fresh environment instances

    Parameters:
        env_name (string):  Name of the environment to load and run
    """
    directory_to_load_from = 'saved_models/' + env_name + '/'
    if not os.path.exists(directory_to_load_from):
        print('Trained model for ' + env_name + ' doesn\'t exist. Run train.py first. Program is exiting now...')
        exit(0)
    imported_meta = tf.train.import_meta_graph(directory_to_load_from + 'final.meta')
    sess = tf.Session()
    imported_meta.restore(sess, tf.train.latest_checkpoint(directory_to_load_from))
    graph = tf.get_default_graph()
    scaled_observation_node = graph.get_tensor_by_name('obs:0')
    output_action_node = graph.get_tensor_by_name('output_action:0')
    scale, offset = get_scale_and_offset(env_name)
    env = gym.make(env_name)
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    aigym_path = os.path.join('/tmp', env_name, now)
    env = wrappers.Monitor(env, aigym_path, video_callable=False, force=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.set_random_seed(args.seed)

    trajectories = apply_learnt_policy(episodes, sess, env, output_action_node, scaled_observation_node, offset, scale, pane=pane)

    # we also cast the tensor to float 16
    #pxobs_t = torch.cat([torch.from_numpy(e['px_obs']) for e in trajectories], 0)
    #actions_t = torch.cat([torch.from_numpy(e['actions']) for e in trajectories], 0).type(torch.float16)
    #rewards_t = torch.cat([torch.from_numpy(e['rewards']) for e in trajectories], 0).type(torch.float16)

    pxobs = np.concatenate([(e['px_obs']) for e in trajectories], 0)
    actions = np.concatenate([(e['actions']) for e in trajectories], 0).astype(np.float16)
    rewards = np.concatenate([(e['rewards']) for e in trajectories], 0).astype(np.float16)

    # save to folder
    savepath = Path(args.save_folder)
    if not Path.exists(savepath):
        os.makedirs(savepath)

    if args.seed == 0:
        filename = 'train.h5'
    elif args.seed == 1:
        filename = 'test.h5'
    elif args.seed == 2:
        filename = 'val.h5'
    else:
        filename = 'px_color:{}_shape:{}_trajs_ep:{}_pane:{}_seed:{}.h5'.format(color, args.shape, episodes, pane or 'both', args.seed)

    with h5py.File(savepath / filename) as h5f:
        h5f.create_dataset('px_obs', data=pxobs)
        h5f.create_dataset('actions', data=actions)
        h5f.create_dataset('rewards', data=rewards)
    print('success')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reload and run trained PPO based policies on OpenAI Gym environment')
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name', default='Reacher-v1')
    parser.add_argument('--n', type=int, default=10, help='N episodes to generate')
    parser.add_argument('--pane', type=str, default=None, help='Generate the target either just on the left pane or the right pane. If not mentioned, default target generation is followed.')
    parser.add_argument('--seed', type=int, default=0, help='For reproducibility')
    parser.add_argument('--save_folder', type=str, default='color/', help='Folder to save the data')
    parser.add_argument('--color', type=str, required=True, help='Color of the target, only for file naming purpose!')
    parser.add_argument('--shape', type=str, required=True, help='Color of the target, only for file naming purpose!')

    args = parser.parse_args()
    create_trajs(args.env_name, args.n, pane=args.pane, color=args.color)
