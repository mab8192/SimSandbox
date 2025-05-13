import numpy as np
import plotly.graph_objects as go

from simmer.driver import Simulation

class HarmonicOscillator(Simulation):
    def __init__(self, k=1.0, m=1.0, x0=1.0, v0=0.0, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.m = m
        self.x0 = x0
        self.v0 = v0

    def _init_state(self):
        self.x = self.x0
        self.v = self.v0

    def _step(self):
        # Calculate acceleration
        a = -self.k * self.x / self.m

        # Update position and velocity
        self.v += a * self.dt
        self.x += self.v * self.dt

    def _get_state(self):
        return np.array([self.x, self.v])

    def plot(self):
        if not self.states:
            raise RuntimeError("No states to plot. Run the simulation first.")

        labels = ["Position", "Velocity"]

        data = np.array(self.states).T
        fig = go.Figure()
        for i, series in enumerate(data):
            fig.add_trace(go.Scatter(x=self.times, y=series, mode='lines', name=labels[i]))
        fig.update_layout(title='Simulation Results', xaxis_title='Time', yaxis_title='State Value')
        fig.show()