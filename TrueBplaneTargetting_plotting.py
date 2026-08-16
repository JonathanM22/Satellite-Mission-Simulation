import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('mission_data.pkl', 'rb') as f:
    d = pickle.load(f)

def add_sphere(ax, radius, color, label):
    u_, v_ = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius*np.cos(u_)*np.sin(v_)
    y = radius*np.sin(u_)*np.sin(v_)
    z = radius*np.cos(v_)
    ax.plot_surface(x, y, z, color=color, alpha=0.3)
    ax.scatter([0],[0],[0], color=color, s=40, label=label)

# --- Plot 1: Earth parking orbit, 1 period pre-burn ---
fig1 = plt.figure(figsize=(8,7)); ax1 = fig1.add_subplot(111, projection='3d')
ep = d['earth_parking']
ax1.plot(*ep['r'].T, color='tab:blue', label='Parking orbit (1 period)')
ax1.scatter(*ep['r'][0], color='black', marker='x', s=80, label='TLI burn point')
add_sphere(ax1, 6378, 'tab:cyan', 'Earth')
ax1.set_title('Earth Parking Orbit — 1 Period to Burn'); ax1.legend(); ax1.set_box_aspect([1,1,1])

# --- Plot 2: Departure, Earth-centered, to SOI ---
fig2 = plt.figure(figsize=(8,7)); ax2 = fig2.add_subplot(111, projection='3d')
l1 = d['leg1_earth_centered']
ax2.plot(*l1['r'].T, color='tab:orange', label='Departure to Earth SOI')
add_sphere(ax2, 6378, 'tab:cyan', 'Earth')
ax2.set_title('Leg 1: Earth Departure'); ax2.legend(); ax2.set_box_aspect([1,1,1])

# --- Plot 3: Heliocentric transfer ---
fig3 = plt.figure(figsize=(9,8)); ax3 = fig3.add_subplot(111, projection='3d')
l2 = d['leg2_heliocentric']
ax3.plot(*l2['r'].T, color='tab:green', label='Heliocentric transfer')
ax3.scatter(*l2['r'][0], color='tab:blue', s=60, label='Earth SOI exit')
ax3.scatter(*l2['r'][-1], color='tab:red', s=60, label='Mars SOI entry')
add_sphere(ax3, 3e6, 'gold', 'Sun')   # oversized for visibility, not to scale
ax3.set_title('Leg 2: Heliocentric Transfer'); ax3.legend(); ax3.set_box_aspect([1,1,1])

# --- Plot 4: Mars-centered incoming ---
fig4 = plt.figure(figsize=(8,7)); ax4 = fig4.add_subplot(111, projection='3d')
l3 = d['leg3_mars_centered']
ax4.plot(*l3['r'].T, color='tab:red', label='Incoming hyperbola')
add_sphere(ax4, 3396, 'tab:orange', 'Mars')
ax4.set_title('Leg 3: Mars Approach'); ax4.legend(); ax4.set_box_aspect([1,1,1])

# --- Plot 5: Capture burn + 1 period after ---
fig5 = plt.figure(figsize=(8,7)); ax5 = fig5.add_subplot(111, projection='3d')
cap = d['capture_orbit']; kp = d['key_points']
ax5.plot(*cap['r'].T, color='tab:purple', label='Parking orbit (post-MOI)')
ax5.scatter(*kp['r_peri'], color='black', marker='x', s=80, label='MOI burn point')
add_sphere(ax5, 3396, 'tab:orange', 'Mars')
ax5.set_title(f"Post-MOI Parking Orbit (a={cap['a']:.1f} km, T={cap['T']/3600:.2f} hr)")
ax5.legend(); ax5.set_box_aspect([1,1,1])

plt.show()