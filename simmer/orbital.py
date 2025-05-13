import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import solve_ivp

from simmer.driver import Simulation


class Orbital(Simulation):
    def __init__(self,
                 r0=np.array([7000.0, 0.0, 0.0]),
                 v0=np.array([0.0, 7.5, 1.0]),
                 mu=398600.4418,
                 R_e=6378.137,
                 J2=8.08262668e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.R_e = R_e
        self.J2 = J2
        self.r0 = np.array(r0, dtype=float)
        self.v0 = np.array(v0, dtype=float)

    def _init_state(self):
        self.state = np.concatenate((self.r0, self.v0))

    def _j2_accel(self, r_vec):
        x, y, z = r_vec
        r = np.linalg.norm(r_vec)
        zx = z / r
        factor = 1.5 * self.J2 * self.mu * self.R_e**2 / r**5
        ax = x * (1 - 5 * zx**2)
        ay = y * (1 - 5 * zx**2)
        az = z * (3 - 5 * zx**2)
        return -self.mu * r_vec / r**3 + factor * np.array([ax, ay, az])

    def _dynamics(self, t, state):
        r = state[:3]
        v = state[3:]
        a = self._j2_accel(r)
        return np.concatenate((v, a))

    def _step(self):
        sol = solve_ivp(
            fun=self._dynamics,
            t_span=(self.time, self.time + self.dt),
            y0=self.state,
            method='RK45',
            rtol=1e-9,
            atol=1e-12
        )
        self.state = sol.y[:, -1]

    def _get_state(self):
        return self.state.copy()

    def plot(self):
        if not self.states:
            raise RuntimeError("No states to plot. Run the simulation first.")

        data = np.array(self.states).T
        times = self.times
        pos_labels = ["x", "y", "z"]
        vel_labels = ["vx", "vy", "vz"]

        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5],
            specs=[
                [{"type": "xy"}, {"type": "scene", "rowspan": 2}],
                [{"type": "xy"}, None]
            ],
            horizontal_spacing=0.1,
            vertical_spacing=0.15,
            subplot_titles=[
                "Position vs Time", "",  # row 1
                "Velocity vs Time", ""   # row 2
            ]
        )

        # Position: row 1, col 1
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data[i],
                    mode='lines',
                    name=pos_labels[i],
                    showlegend=True
                ),
                row=1,
                col=1
            )

        # Velocity: row 2, col 1
        for i in range(3, 6):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=data[i],
                    mode='lines',
                    name=vel_labels[i - 3],
                    showlegend=True
                ),
                row=2,
                col=1
            )

        # Initial 3D frame
        fig.add_trace(
            go.Scatter3d(
                x=data[0],
                y=data[1],
                z=data[2],
                mode='markers+lines',
                marker=dict(size=2),
                line=dict(width=1),
                name="Trajectory",
                showlegend=True
            ),
            row=1,
            col=2
        )

        # Animation frames
        frame_step = max(1, len(times) // 100)
        frames = [
            go.Frame(
                data=[go.Scatter3d(
                    x=data[0][:k],
                    y=data[1][:k],
                    z=data[2][:k],
                    mode='markers+lines',
                    marker=dict(size=2),
                    line=dict(width=1),
                    name="Trajectory",
                    showlegend=False
                )],
                name=str(k)
            )
            for k in range(2, len(times)+1, frame_step)
        ]
        fig.frames = frames

        # Calculate bounds for equal aspect ratio
        padding = 0.05
        x, y, z = data[0], data[1], data[2]
        x_range = [x.min(), x.max()]
        y_range = [y.min(), y.max()]
        z_range = [z.min(), z.max()]
        min_bound = min(x_range[0], y_range[0], z_range[0])
        max_bound = max(x_range[1], y_range[1], z_range[1])
        span = (max_bound - min_bound) * (1 + padding)
        center = (max_bound + min_bound) / 2

        fig.update_scenes(
            dict(
                xaxis=dict(title='X [km]', range=[center - span/2, center + span/2]),
                yaxis=dict(title='Y [km]', range=[center - span/2, center + span/2]),
                zaxis=dict(title='Z [km]', range=[center - span/2, center + span/2]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Orbital Simulation Results (Animated)",
            showlegend=True,
            updatemenus=[{
                "type": "buttons",
                "direction": "right",
                "x": 0.73,  # Top-right corner over 3D plot
                "y": 0.95,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                    },
                    {
                        "label": "Restart",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": False}]
                    },
                ]
            }]
        )

        fig.show()

