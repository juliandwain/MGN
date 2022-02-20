import pathlib
import pickle
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import tri as mtri
import xarray
import numpy as np

rollout_fname = pathlib.Path("./mgn_output/prediction/rollout.pkl")
with rollout_fname.open(mode="rb") as f:
    rollout_data = pickle.load(f)
trajectory_fname = pathlib.Path("trajectory.pkl")
with trajectory_fname.open(mode="rb") as f:
    trajectory_data = pickle.load(f)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
skip = 10
num_steps = rollout_data["pred"].shape[0]
num_frames = len(rollout_data) * num_steps // skip

# compute bounds
bounds = []
for trajectory in rollout_data["pred"]:
    bb_min = trajectory.min(axis=(0, 1))
    bb_max = trajectory.max(axis=(0, 1))
    bounds.append((bb_min, bb_max))


def animate(num):
    step = (num * skip) % num_steps
    traj = (num * skip) // num_steps
    ax.cla()
    ax.set_aspect("equal")
    ax.set_axis_off()
    vmin, vmax = bounds[traj]
    pos = trajectory_data["mesh_pos"][traj]
    faces = trajectory_data["cells"][step]
    velocity = rollout_data["pred"][step]
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    ax.tripcolor(triang, velocity[:, 0], vmin=vmin, vmax=vmax)
    ax.triplot(triang, "ko-", ms=0.5, lw=0.3)
    ax.set_title("Trajectory %d Step %d" % (traj, step))
    return (fig,)


animation_fname = "./animation.gif"
# writer = animation.FFMpegWriter(fps=60)
writer = animation.PillowWriter(fps=30)
anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
anim.save(animation_fname, writer=writer)
plt.show(block=True)
