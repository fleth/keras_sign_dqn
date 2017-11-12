# -*- coding: utf-8 -*-
import numpy
import utils

class Env:
    def __init__(self):
        self.terminal = False
        self.step = 0
        self.state_size = 25
        self.state = numpy.zeros((self.state_size, self.state_size))
        self.wav = self._sign(self.state_size)
        self.max_step = len(self.wav)
        self.reward = 0
        self.rewards = utils.rewards(self.wav)
        self.actions = (0, 1)
        print("[Env] max_step: %s" % self.max_step)

    def _sign(self, a=1):
        fs = 1000 #サンプリング周波数
        f0 = 100  #周波数
        wav = []
        sec = 5 #秒
        for n in numpy.arange(fs * sec):
            s = a * numpy.sin(2.0 * numpy.pi * f0 * n / fs)
            wav.append(s)
        return wav

    def update(self, action):
        self.step += 1

        sign = self.rewards[self.step + self.state_size]

        self.reward = 0

        if action == 1:
            if sign < 0.5:
                self.reward = -1
            elif sign > 0.5:
                self.reward = 1
        else:
            if sign > 0.5:
                self.reward = -1
            elif sign < 0.5:
                self.reward = 1

        begin = self.step
        end = self.step + self.state_size
        self.state = self.screen(self.wav[begin:end])
        self.terminal = False
        if self.max_step-1 <= end:
            self.terminal = True
            self.reward = 1
        if self.reward < 0:
            self.terminal = True

    def execute_action(self, action):
        self.update(action)

    def screen(self, wave):
        screen = numpy.zeros((self.state_size, self.state_size))

        for i,w in enumerate(wave):
            screen[i, int(w)] = 1

        return screen

    def observe(self):
        return self.state, self.reward, self.terminal

    def reset(self):
        self.step = 0
        self.reward = 0
        self.terminal = False
        self.state = numpy.zeros((self.state_size, self.state_size))
