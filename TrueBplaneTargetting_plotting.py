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

#%%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.time import Time

with open('mission_data.pkl', 'rb') as f:
    d = pickle.load(f)

EARTH_R = 6378.0
MARS_R = 3396.0
SUN_R = 3e6   # exaggerated for visibility, not to scale

def jd_to_str(jd):
    return Time(jd, format='jd').iso[:19] + ' UTC'

def add_sphere(ax, radius, color):
    u_, v_ = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius*np.cos(u_)*np.sin(v_)
    y = radius*np.sin(u_)*np.sin(v_)
    z = radius*np.cos(v_)
    ax.plot_surface(x, y, z, color=color, alpha=0.35, linewidth=0)

def set_zoom(ax, half_range, center=(0,0,0)):
    cx, cy, cz = center
    ax.set_xlim3d(cx-half_range, cx+half_range)
    ax.set_ylim3d(cy-half_range, cy+half_range)
    ax.set_zlim3d(cz-half_range, cz+half_range)

# ============================================================
# PHASE 1: Earth parking orbit -> departure burn -> Earth SOI
# Starts zoomed on the parking orbit, zooms OUT once the burn happens.
# ============================================================
def animate_phase1(d):
    ep = d['earth_parking']
    l1 = d['leg1_earth_centered']

    # skip l1's first point since it duplicates ep's last point (same
    # position at the burn instant, only velocity changes) -- avoids a
    # one-frame stall in the animation right at the burn
    r_all = np.vstack([ep['r'], l1['r'][1:]])
    t_all = np.concatenate([ep['t_jd'], l1['t_jd'][1:]])
    burn_idx = len(ep['r'])

    zoom_near = 3.5 * ep['a']
    zoom_far  = 1.15 * d['key_points']['earth_soi']

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    add_sphere(ax, EARTH_R, 'tab:cyan')
    line,  = ax.plot([], [], [], color='tab:blue', lw=1.5)
    point, = ax.plot([], [], [], 'o', color='black', ms=5)
    time_text  = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                            bbox=dict(facecolor='white', alpha=0.75))
    phase_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='tab:red')
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]'); ax.set_zlabel('Z [km]')
    ax.set_title('Phase 1: Earth Parking Orbit -> Departure -> Earth SOI')

    def update(i):
        line.set_data(r_all[:i+1, 0], r_all[:i+1, 1])
        line.set_3d_properties(r_all[:i+1, 2])
        point.set_data([r_all[i, 0]], [r_all[i, 1]])
        point.set_3d_properties([r_all[i, 2]])

        if i < burn_idx:
            half_range = zoom_near
            phase_text.set_text('Phase: Coasting in parking orbit')
        else:
            frac = (i - burn_idx) / max(1, len(r_all) - 1 - burn_idx)
            half_range = zoom_near + frac * (zoom_far - zoom_near)
            phase_text.set_text('Phase: Departure burn -> Earth SOI')

        set_zoom(ax, half_range)
        time_text.set_text(jd_to_str(t_all[i]))
        return line, point, time_text, phase_text

    return FuncAnimation(fig, update, frames=len(r_all), interval=40, repeat=False)

# ============================================================
# PHASE 2: Heliocentric transfer, Sun/Earth/Mars all shown, zoomed out
# ============================================================
def animate_phase2(d):
    l2 = d['leg2_heliocentric']
    et = d['earth_track']
    mt = d['mars_track']
    r_sc, t_sc = l2['r'], l2['t_jd']
    r_earth, r_mars = et['r'], mt['r']

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    add_sphere(ax, SUN_R, 'gold')

    sc_line,    = ax.plot([], [], [], color='tab:green', lw=1.5, label='Spacecraft')
    sc_point,   = ax.plot([], [], [], 'o', color='tab:green', ms=6)
    earth_line, = ax.plot([], [], [], color='tab:cyan', lw=1, ls='--', label='Earth')
    earth_point,= ax.plot([], [], [], 'o', color='tab:cyan', ms=6)
    mars_line,  = ax.plot([], [], [], color='tab:red', lw=1, ls='--', label='Mars')
    mars_point, = ax.plot([], [], [], 'o', color='tab:red', ms=6)
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                           bbox=dict(facecolor='white', alpha=0.75))
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]'); ax.set_zlabel('Z [km]')
    ax.set_title('Phase 2: Heliocentric Transfer (Earth -> Mars)')
    ax.legend(loc='upper right')

    max_extent = 1.15 * max(np.max(np.abs(r_sc)), np.max(np.abs(r_earth)), np.max(np.abs(r_mars)))
    set_zoom(ax, max_extent)

    def update(i):
        sc_line.set_data(r_sc[:i+1,0], r_sc[:i+1,1]); sc_line.set_3d_properties(r_sc[:i+1,2])
        sc_point.set_data([r_sc[i,0]], [r_sc[i,1]]); sc_point.set_3d_properties([r_sc[i,2]])

        earth_line.set_data(r_earth[:i+1,0], r_earth[:i+1,1]); earth_line.set_3d_properties(r_earth[:i+1,2])
        earth_point.set_data([r_earth[i,0]], [r_earth[i,1]]); earth_point.set_3d_properties([r_earth[i,2]])

        mars_line.set_data(r_mars[:i+1,0], r_mars[:i+1,1]); mars_line.set_3d_properties(r_mars[:i+1,2])
        mars_point.set_data([r_mars[i,0]], [r_mars[i,1]]); mars_point.set_3d_properties([r_mars[i,2]])

        time_text.set_text(jd_to_str(t_sc[i]))
        return sc_line, sc_point, earth_line, earth_point, mars_line, mars_point, time_text

    return FuncAnimation(fig, update, frames=len(r_sc), interval=40, repeat=False)

# ============================================================
# PHASE 3: Mars approach (zoom IN) -> MOI burn -> 1 period of parking orbit
# ============================================================
def animate_phase3(d):
    l3  = d['leg3_mars_centered']
    cap = d['capture_orbit']

    # skip cap's first point since it duplicates l3's last point (same
    # position at periapsis, only velocity changes at the MOI burn)
    r_all = np.vstack([l3['r'], cap['r'][1:]])
    t_all = np.concatenate([l3['t_jd'], cap['t_jd'][1:]])
    moi_idx = len(l3['r'])

    zoom_far  = 1.15 * d['key_points']['mars_soi']
    zoom_near = 3.5 * cap['a']

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    add_sphere(ax, MARS_R, 'tab:orange')
    line,  = ax.plot([], [], [], color='tab:red', lw=1.5)
    point, = ax.plot([], [], [], 'o', color='black', ms=5)
    time_text  = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                            bbox=dict(facecolor='white', alpha=0.75))
    phase_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='tab:purple')
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]'); ax.set_zlabel('Z [km]')
    ax.set_title('Phase 3: Mars Approach -> MOI Burn -> Parking Orbit')

    def update(i):
        line.set_data(r_all[:i+1, 0], r_all[:i+1, 1])
        line.set_3d_properties(r_all[:i+1, 2])
        point.set_data([r_all[i, 0]], [r_all[i, 1]])
        point.set_3d_properties([r_all[i, 2]])

        if i < moi_idx:
            frac = i / max(1, moi_idx - 1)
            half_range = zoom_far + frac * (zoom_near - zoom_far)   # zoom in
            phase_text.set_text('Phase: Approaching Mars SOI')
        else:
            half_range = zoom_near
            phase_text.set_text('Phase: Post-MOI Parking Orbit')

        set_zoom(ax, half_range)
        time_text.set_text(jd_to_str(t_all[i]))
        return line, point, time_text, phase_text

    return FuncAnimation(fig, update, frames=len(r_all), interval=40, repeat=False)

# --- Run all three ---
# IMPORTANT: keep references to the animations (anim1/2/3) -- if they go out
# of scope Python garbage-collects them and the animation just freezes.
anim1 = animate_phase1(d)
anim2 = animate_phase2(d)
anim3 = animate_phase3(d)
plt.show()