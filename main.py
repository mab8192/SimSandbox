from simmer import *

def run_harmonic():
    sim = HarmonicOscillator(tf=20, dt=0.01, x0=1.0, v0=0.0)
    sim.run()
    sim.plot()


def run_orbital():
    sim = Orbital(
        r0=[7000, 700, 0],
        v0=[0, 7.5, 1.0],
        tf=86400,  # 1 day
        dt=60.0    # 60s time step
    )
    sim.run()
    sim.plot()


if __name__ == "__main__":
    run_orbital()
