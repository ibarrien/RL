import argparse
import sys
import os
from pathlib import Path
from itertools import product

import gym
from gym import wrappers, logger

import numpy as np
import time
import copy
import random
import statistics

from collections import defaultdict

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from tiling import Tiling

import random

from semigradsarsa import SemiGradientSarsa
            
if __name__ == '__main__':
    # SEED = 28
    SEED = 29
    random.seed(SEED)
    parser = argparse.ArgumentParser(description=None)
    ENVIRONMENT = 'MountainCar-v0'
    parser.add_argument('env_id', nargs='?', default=ENVIRONMENT, help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)
    # env.metadata['video.frames_per_second'] = 24
    # print(env.metadata)
    # input()
    # rec = VideoRecorder(env, path='./video/output01.mp4')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    # env.seed(0)
    env.seed(1)
    TIME_BUDGET = 5
    ITERATION_BUDGET = 4000
    LOOKAHEAD_TARGET = 50
    EPSILON = 0
    ALPHA = 0.5/8
    GAMMA = 1
    REGULARIZATION = 0
    
    HACK_EPSILON = 0.0000001 # Hack to deal with half open intervals
    N_TILINGS = 8
    bounds = [[-1.2, 0.6 + HACK_EPSILON, 8], [-0.7, 0.7 + HACK_EPSILON, 8]]
    tl = Tiling(bounds, N_TILINGS)
    def feature6(x):
        # print(np.sum(tl.pt_to_feats(x)))
        # input()
        return tl.pt_to_feats(x)

    DEBUG = False
    agent = SemiGradientSarsa(env.action_space, env.observation_space, REGULARIZATION, feature6)

    episode_count = 100000
    RECORDING_INTERVAL = 50
    TIMESTR = time.strftime("%Y%m%d-%H%M%S")
    RECORDER_PATH = Path('./video/' + TIMESTR)
    os.mkdir(RECORDER_PATH)

    reward = 0
    done = False

    rec = VideoRecorder(env, path=str(RECORDER_PATH / ('video' + '.mp4')))

    for i in range(episode_count):
        
        ob = env.reset()
        action = agent.act(ob, EPSILON)
        if i % RECORDING_INTERVAL == 0:
            pass
            # rec = VideoRecorder(env, path=str(RECORDER_PATH / (f'{i:06d}' + '.mp4')))
        try:
            sum_reward = 0
            while True:
                # print("################")
                
                if i % RECORDING_INTERVAL == 0:
                    # print("### 301", agent.weights)
                    print("### 302", env.state)
                    print("### 303 episode: ", i)
                    env.render()
                    rec.capture_frame()
                # print("### 201 action: ", action)
                ob_prime, reward, done, _ = env.step(action)
                if done:
                    agent.terminal_update(ALPHA, action, ob, reward)
                    if i % RECORDING_INTERVAL == 0:
                        # rec.close()
                        pass
                    break
                    continue
                action_prime = agent.act(ob_prime, EPSILON)
                agent.update(ALPHA, GAMMA, action, ob, reward, ob_prime, action_prime, i)
                ob = ob_prime
                action = action_prime
                # print("### 101 observed state: ", ob)
                sum_reward += reward
                # print("### 008 sum_reward: ", sum_reward)
                # if done:
                #     rec.close()
                #     break
        except KeyboardInterrupt as e:
            rec.close()
            env.close()
            raise e

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        print("### 303 episode: ", i, " total reward: ", sum_reward)

    # Close the env and write monitor result info to disk
        # print("### sum reward: ", sum_reward)

    rec.close()
           
    env.close()
