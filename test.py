# -*- coding: utf-8 -*-
import argparse
import numpy
import utils
import json
from env import Env
from agent import Agent

env = Env()
agent = Agent(env.actions)
agent.load_model()

terminal = False
total_frame = 0
max_step = 0
frame = 0
env.reset()
state_t, reward_t, terminal = env.observe()
while not terminal:

    action_t, is_random = agent.select_action([state_t], 0.0)
    env.execute_action(action_t)

    state_t, reward_t, terminal = env.observe()

    frame += 1
    total_frame += 1
    if max_step < env.step:
        max_step = env.step

    print("frame: %s, total_frame: %s, terminal: %s, action: %s, reward: %s" % (frame, total_frame, terminal, action_t, reward_t))

print("max_step: %s" % max_step)
