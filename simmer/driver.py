import numpy as np
import plotly.graph_objects as go
from abc import ABC, abstractmethod

class Simulation(ABC):
    def __init__(self, t0=0.0, tf=10.0, dt=0.01, log_states=True):
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.time = t0
        self.states = []
        self.times = []
        self.log_states = log_states

    def run(self):
        self.reset()
        while self.time < self.tf:
            self._step()
            self.time += self.dt
            if self.log_states:
                self._log_state()

    def reset(self):
        self.time = self.t0
        self.states.clear()
        self.times.clear()
        self._init_state()

    def _log_state(self):
        self.states.append(np.copy(self._get_state()))
        self.times.append(self.time)

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def _init_state(self):
        pass

    @abstractmethod
    def _step(self):
        pass

    @abstractmethod
    def _get_state(self):
        pass
