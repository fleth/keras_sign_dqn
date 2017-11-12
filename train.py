# -*- coding: utf-8 -*-
import argparse
import numpy
import utils
import json
from env import Env
from agent import Agent

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", dest="load", action="store_true",
    default=False, help='Load trained model (default: off)')
args = parser.parse_args()

env = Env()
agent = Agent(env.actions)

if args.load:
    agent.load_model()

terminal = False
n_epochs = 5000
e = 0
total_frame = 0
do_replay_count = 0
max_step = 0

while e < n_epochs:
    frame = 0
    loss = 0.0
    Q_max = 0.0
    env.reset()
    state_t_1, reward_t, terminal = env.observe()
    while not terminal:
        state_t = state_t_1

        action_t, is_random = agent.select_action([state_t], agent.exploration)
        env.execute_action(action_t)

        state_t_1, reward_t, terminal = env.observe()

        start_replay = False
        start_replay = agent.store_experience([state_t], action_t, reward_t, [state_t_1], terminal)

        if start_replay:
            do_replay_count += 1
            agent.update_exploration(e)
            if do_replay_count > 2:
                agent.replay()
                do_replay_count = 0

        if total_frame % 500 == 0 and start_replay:
            agent.update_target_model()

        frame += 1
        total_frame += 1
        loss += agent.current_loss
        Q_max += numpy.max(agent.Q_values([state_t]))


        if start_replay:
            agent.replay()
            print("epochs: %s/%s, loss: %s, Q_max: %s, terminal: %s, step: %s, action: %s, reward: %s, random: %s" % (e, n_epochs, loss / frame, Q_max / frame, terminal, env.step, action_t, reward_t, is_random))
            e += 1
            if max_step < env.step:
                max_step = env.step
        else:
            print("frame: %s, total_frame: %s, terminal: %s, action: %s, reward: %s" % (frame, total_frame, terminal, action_t, reward_t))

agent.save_model()
print("max_step: %s" % max_step)
