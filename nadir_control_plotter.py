# Custom libs
from orbit import *
from Orbit_util import *
from body import *
from Universal_Variable import *
from quaternion import *
from reactionwh import *

# Standard libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib import gridspec
import time

data = np.load("nadir_control_sim_data.npy", allow_pickle=True)[()]

r_hist = data["r_hist"]
v_hist = data["v_hist"]
q_sat_hist = data["q_sat_hist"]
sat_w_hist = data["sat_w_hist"]
q_error_hist = data["q_error_hist"]
L_hist = data["L_hist"]
lvlh_attitude_hist = data["lvlh_attitude_hist"]

plot_error = True
if plot_error:
    sti = 0
    eni = -1
    """PLOT Q_ERROR"""
    print(len(q_error_hist))
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    sti = 0
    eni = 1000
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(q_error_hist[sti:eni, 0], color='tab:red', label='q1')
    ax1.plot(q_error_hist[sti:eni, 1], color='tab:green', label='q2')
    ax1.plot(q_error_hist[sti:eni, 2], color='tab:blue', label='q3')
    ax1.plot(q_error_hist[sti:eni, 3], color='tab:orange', label='q4')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("Quaternion Error")
    ax1.set_title("Quaternion Error vs steps")
    ax1.legend(loc='right')
    ax1.grid(True)

    """PLOT SAT_W"""
    ax1 = plt.subplot(3, 1, 2)
    ax1.plot(sat_w_hist[sti:eni, 0], color='tab:red', label='w1')
    ax1.plot(sat_w_hist[sti:eni, 1], color='tab:green', label='w2')
    ax1.plot(sat_w_hist[sti:eni, 2], color='tab:blue', label='w3')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("sat_w")
    ax1.set_title("Sat Angular Velocity vs steps")
    ax1.legend(loc='right')
    ax1.grid(True)

    """PLOT CONTROL MOMENT"""
    ax1 = plt.subplot(3, 1, 3)
    ax1.plot(L_hist[sti:eni, 0], color='tab:red', label='w1')
    ax1.plot(L_hist[sti:eni, 1], color='tab:green', label='w2')
    ax1.plot(L_hist[sti:eni, 2], color='tab:blue', label='w3')
    ax1.set_xlabel("steps")
    ax1.set_ylabel("L")
    ax1.set_title("Control Moment vs time")
    ax1.legend(loc='right')
    ax1.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.show()

plot_vid = False
""" ----------------- PLOT CHANGING FRAME ----------------- """
if plot_vid:
    # Animation driver
    # Change to reflect your file location!
    plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\jonam\Desktop\Aero-Project\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

    # Vars
    sti = 0
    eni = 100
    stp = 1
    dpi = 150
    fps = 24
    pause = 2

    # Setup Ani
    metadata = dict(title='sat_orientation', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    frames = range(sti, eni, stp)

    """NEED TO SAVE IT OUTSIDE OF REPO TOO BIG FOR GIT"""
    save_path = r"C:\Users\jonam\Desktop\Aero-Project\nadir_control_sim.mp4"
    with writer.saving(fig, "nadir_control_sim.mp4", dpi):
        for i, f in enumerate(frames):
            plt.cla()

            lvlh_attitude = lvlh_attitude_hist[f]
            q_sat_attitude = Quaternion(
                q_sat_hist[f].reshape(4, 1)).to_attitude()

            """ PLOT LVLH FRAME """
            origin = np.array([0, 0, 0])
            O1 = lvlh_attitude[:, 0]
            O2 = lvlh_attitude[:, 1]
            O3 = lvlh_attitude[:, 2]
            lvlh_color = "tab:blue"

            ax.quiver(origin[0], origin[1], origin[2],
                      O1[0], O1[1], O1[2],
                      color=lvlh_color, label='O1')

            ax.quiver(origin[0], origin[1], origin[2],
                      O2[0], O2[1], O2[2],
                      color=lvlh_color, label='O2')

            ax.quiver(origin[0], origin[1], origin[2],
                      O3[0], O3[1], O3[2],
                      color=lvlh_color, label='O3')

            """ PLOT SAT ORIENTATION"""
            S1 = q_sat_attitude[:, 0]
            S2 = q_sat_attitude[:, 1]
            S3 = q_sat_attitude[:, 2]
            q_sat_color = "tab:orange"

            ax.quiver(origin[0], origin[1], origin[2],
                      S1[0], S1[1], S1[2],
                      color=q_sat_color, label='S1')

            ax.quiver(origin[0], origin[1], origin[2],
                      S2[0], S2[1], S2[2],
                      color=q_sat_color, label='S2')

            ax.quiver(origin[0], origin[1], origin[2],
                      S3[0], S3[1], S3[2],
                      color=q_sat_color, label='S3')

            ax.set_title(f"Quaternion Control", fontsize=14, pad=10)
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_zlim([-1.5, 1.5])
            plt.tight_layout()

            # ax.view_init(elev=45, azim=azim_angle[i], roll=0)
            ax.legend(loc='best')
            writer.grab_frame()
            print(f'Generated Leg 1 frame {i} of {len(frames)}')

    print("Done generating Video")

""" ---------------------- CHAT GPT HELPED WITH GRID SPEC ----------------------  """
# Figure and gridspec
sti, eni, stp = 0, 1000, 2
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1.2], height_ratios=[
                       1, 1, 1], wspace=0.3, hspace=0.4)

# Left column: stacked plots
ax_qerror = fig.add_subplot(gs[0, 0])
ax_satw = fig.add_subplot(gs[1, 0])
ax_L = fig.add_subplot(gs[2, 0])

# Right column: 3D animation
ax_3d = fig.add_subplot(gs[:, 1], projection='3d')  # spans all 3 rows

plot_vid = False
if plot_vid:
    plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\jonam\Desktop\Aero-Project\ffmpeg-master-latest-win64-gpl\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

    metadata = dict(title='sat_orientation', artist='jonamat03@gmail.com')
    writer = FFMpegWriter(fps=24, metadata=metadata)

    frames = range(sti, eni, stp)
    azim_angle = np.linspace(0, 90, len(frames))
    origin = np.array([0, 0, 0])
    L_arrow = 1.0

    """NEED TO SAVE IT OUTSIDE OF REPO TOO BIG FOR GIT"""
    save_path = r"C:\Users\jonam\Desktop\Aero-Project\nadir_control_sim_combined.mp4"
    with writer.saving(fig, save_path, dpi=150):
        for i, f in enumerate(frames):
            ax_3d.cla()  # clear 3D axes
            ax_qerror.cla()
            ax_satw.cla()
            ax_L.cla()

            # Q_ERROR
            ax_qerror.plot(q_error_hist[sti:f, 0], label='q1')
            ax_qerror.plot(q_error_hist[sti:f, 1], label='q2')
            ax_qerror.plot(q_error_hist[sti:f, 2], label='q3')
            ax_qerror.plot(q_error_hist[sti:f, 3], label='q4')
            ax_qerror.set_ylabel("Quaternion Error")
            ax_qerror.set_title("Quaternion Error vs steps")
            ax_qerror.legend()
            ax_qerror.grid(True)

            # SAT_W
            ax_satw.plot(sat_w_hist[sti:f, 0], label='w1')
            ax_satw.plot(sat_w_hist[sti:f, 1], label='w2')
            ax_satw.plot(sat_w_hist[sti:f, 2], label='w3')
            ax_satw.set_ylabel("sat_w")
            ax_satw.set_title("Satellite Angular Velocity")
            ax_satw.legend()
            ax_satw.grid(True)

            # CONTROL MOMENT
            ax_L.plot(L_hist[sti:f, 0], label='L1')
            ax_L.plot(L_hist[sti:f, 1], label='L2')
            ax_L.plot(L_hist[sti:f, 2], label='L3')
            ax_L.set_xlabel("steps")
            ax_L.set_ylabel("L")
            ax_L.set_title("Control Moment")
            ax_L.legend()
            ax_L.grid(True)

            # LVLH frame
            lvlh_attitude = lvlh_attitude_hist[f]
            O1, O2, O3 = lvlh_attitude[:,
                                       0], lvlh_attitude[:, 1], lvlh_attitude[:, 2]
            ax_3d.quiver(*origin, *O1, color='tab:blue',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *O2, color='tab:blue',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *O3, color='tab:blue',
                         length=L_arrow, normalize=True)

            # Satellite frame
            q_sat_attitude = Quaternion(
                q_sat_hist[f].reshape(4, 1)).to_attitude()
            S1, S2, S3 = q_sat_attitude[:,
                                        0], q_sat_attitude[:, 1], q_sat_attitude[:, 2]
            ax_3d.quiver(*origin, *S1, color='tab:orange',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *S2, color='tab:orange',
                         length=L_arrow, normalize=True)
            ax_3d.quiver(*origin, *S3, color='tab:orange',
                         length=L_arrow, normalize=True)

            ax_3d.set_xlim([-1.5, 1.5])
            ax_3d.set_ylim([-1.5, 1.5])
            ax_3d.set_zlim([-1.5, 1.5])
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title('Quaternion Control', fontsize=12)
            ax_3d.set_box_aspect([1, 1, 1])
            ax_3d.view_init(elev=20, azim=azim_angle[i])

            writer.grab_frame()
            print(f'Frame {i+1}/{len(frames)} generated')
