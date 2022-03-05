import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from matplotlib.animation import FuncAnimation

plt.gcf().set_dpi(600)
plt.rcParams.update({'axes.titlesize': 'x-small', 'axes.labelsize': 'x-small',
                     'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small',
                     'figure.titlesize': 'small'})


# %%
# Here are some example of state spaces based on MOOG LADDER FILTER GENERALIZATIONS BASED ON STATE VARIABLE FILTERS

def lowpass(wc, k, r, gamma=1):
    return signal.StateSpace(
        wc * np.array([[-2 * r, 1, 0, 4 * k * r ** 2], [-1, 0, 0, 0], [0, -1, -2 * r, 1], [0, 0, -1, 0]]),
        wc * np.array([[1, 0, 0, 0]]).T,
        np.array([[0, -gamma, 0, 0]]),
        np.array([[0]])
    ).to_tf()


def lowpass2(wc, k, r, gamma=1):
    """This filter has 80db/dec"""
    return signal.StateSpace(
        wc * np.array([[-2 * r, 1, 0, 4 * k * r ** 2], [-1, 0, 0, 0], [0, -1, -2 * r, 1], [0, 0, -1, 0]]),
        wc * np.array([[1, 0, 0, 0]]).T,
        np.array([[0, 0, 0, -gamma]]),
        np.array([[0]])
    ).to_tf()


def bandpass(wc, k, r):
    return signal.StateSpace(
        wc * np.array([[-2 * r, 1, 0, 4 * k * r ** 2], [-1, 0, 0, 0], [0, -1, -2 * r, 1], [0, 0, -1, 0]]),
        wc * np.array([[1, 0, 0, 0]]).T,
        np.array([[0, 1, 2 * r, -1]]),
        np.array([[0]])
    ).to_tf()


def highpass(wc, k, r):
    return signal.StateSpace(
        wc * np.array([[-2 * r, 1, 0, 4 * k * r ** 2], [-1, 0, 0, 0], [0, -1, -2 * r, 1], [0, 0, -1, 0]]),
        wc * np.array([[1, 0, 0, 0]]).T,
        np.array([[-2 * r, 1, 0, 4 * k * r ** 2]]),
        np.array([[1]])
    ).to_tf()


def all_40db_filters(wc, k, r):
    """Combination of -40dB/dec lowpass/bandpass/highpass filters"""
    return signal.StateSpace(
        wc * np.array([[-2 * r, 1, 0, 4 * k * r ** 2], [-1, 0, 0, 0], [0, -1, -2 * r, 1], [0, 0, -1, 0]]),
        wc * np.array([[1, 0, 0, 0]]).T,
        np.array([[0, -gamma, 0, 0], [0, 1, 2 * r, -1], [-2 * r, 1, 0, 4 * k * r ** 2]]),
        np.array([[0], [0], [1]])
    ).to_tf()


# %%
# Bode plot of low-band-highpass filters with 40db/dec

wc, k, r = 1_000, 0.8, 0.9

plt.figure()
for fn in (lowpass, lowpass2, bandpass, highpass):
    w, mag, _ = signal.bode(fn(wc, k, r), np.linspace(0, 2 * np.pi * 44.1e3, 1_000_000))
    plt.semilogx(w, mag, label=fn.__name__)
plt.legend()
plt.hlines(-40, wc / 10, wc * 10, linestyles='dashed')
plt.hlines(-80, wc / 10, wc * 10, linestyles='dashed')
plt.vlines(wc, -100, 20, linestyles='dashed')
plt.vlines(wc / 10, -100, 20, linestyles='dashed')
plt.vlines(wc * 10, -100, 20, linestyles='dashed')
plt.suptitle('Bode magnitude plot')
plt.show()


#%%
# Bode plot with mix between low-band-highpass filters with 40db/dec

def dry_wet(mix):
    assert 0.0 <= mix <= 1.0, f'Expected mix in [0, 1], got {mix:.2f}'
    return 2 * np.array(
        [0.5 - mix, mix, 0] if 0.0 <= mix <= 0.5 else
        [0, 1 - mix, mix - 0.5]
    )


hs = all_40db_filters(wc, k, r)


def bode(mix, ws=np.linspace(0, 2 * np.pi * 44.1e3, 10_000)):
    hs_num = sum(k * y for k, y in zip(dry_wet(mix), hs.num))
    return signal.bode((hs_num, hs.den), ws)


plt.figure()
plt.tight_layout()

for mix in np.linspace(0, 1, 25):
    w, mag, _ = bode(mix)
    plt.semilogx(w, mag, label=f'mix={mix:.2f}')

plt.legend()
plt.suptitle(f'k={k}, r={r}, wc={wc}')
plt.hlines(-40, 1, 100000, linestyles='dashed')
plt.hlines(-80, 1, 100000, linestyles='dashed')
plt.vlines(wc, -100, 20, linestyles='dashed')
plt.vlines(wc / 10, -100, 20, linestyles='dashed')
plt.vlines(wc * 10, -100, 20, linestyles='dashed')
plt.show()

#%%
# Animation of above

fig, axs = plt.subplots(2, 1)
steps = 100


def bode_plot(i):
    for a in axs:
        a.clear()
    mix = i / steps
    w, mag, phase = bode(mix)
    k_lp, k_bp, k_hp = dry_wet(mix)
    plt.suptitle(f'mix={mix:.2f}, lp={k_lp:.2f}, bp={k_bp:.2f}, hp={k_hp:.2f}')
    axs[0].set(xlim=(1, 20_000), ylim=(-100, 20))
    axs[0].semilogx(w, mag)
    axs[1].set(xlim=(1, 20_000), ylim=(-360, 360))
    axs[1].semilogx(w, phase)


anim = FuncAnimation(fig, bode_plot, frames=np.arange(steps))
anim.save(f'bode={steps}_clear_fps=60.mp4', dpi=400, fps=60, writer='ffmpeg')


#%%

from sympy import symbols, tan, eye, simplify, Matrix

B = Matrix([[1, 2, 3, 4]])
C = Matrix([[1, 2, 3, 4]]).T

assert B.shape == (1, 4) and C.shape == (4, 1)

B = np.array([[1, 2, 3, 4]])
C = np.array([[1, 2, 3, 4]]).T

assert B.shape == (1, 4) and C.shape == (4, 1)

B*C

# wc, r, k, T = symbols('\omega_c, r, k, T')

wc, k, r, T, gamma = 0.1, 0.8, 0.9, 1, 1

A = wc * np.array([[-2 * r, 1, 0, 4 * k * r ** 2], [-1, 0, 0, 0], [0, -1, -2 * r, 1], [0, 0, -1, 0]])
B = wc * np.array([[1, 0, 0, 0]]).T
C = np.array([[0, 0, 0, -gamma]])
D = np.array([[0]])

assert A.shape == (4, 4) and B.shape == (4, 1) and C.shape == (1, 4) and D.shape == (1, 1)

A0, B0, C0, D0, _ = signal.cont2discrete((A, B, C, D), dt=T, method='gbt', alpha=0.5)

plt.figure()
w, mag, _ = signal.dbode((A0, B0, C0, D0, T))
plt.semilogx(w, mag)
# w, mag, _ = signal.bode((A, B, C, D))
# plt.semilogx(w, mag)
# plt.show()

g = 2*wc/T
H = g * np.linalg.inv(np.eye(4) - g * A)
I = np.eye(4)

A_tilde = 2 * np.dot(H, A) + I
B_tilde = 2 * np.dot(H, B)
C_tilde = np.dot(C, (np.dot(H, A) + np.eye(4)))
D_tilde = np.dot(C, B)

# A_tilde = np.linalg.inv(np.eye(4) - wc*T/2*A) * wc*T/2 @ A
# B_tilde = np.linalg.inv(np.eye(4) - wc*T/2*A) * wc*T/2 @ B
# C_tilde = C
# D_tilde = D
# C_tilde = np.dot(C, (np.dot(H, A) + np.eye(4)))
# D_tilde = np.dot(C, B)

assert A_tilde.shape == (4, 4) and B_tilde.shape == (4, 1) and C_tilde.shape == (1, 4) and D_tilde.shape == (1, 1)

# plt.figure()
w, mag, _ = signal.dbode((A_tilde, B_tilde, C_tilde, D_tilde, T))
plt.semilogx(w, mag)
plt.show()

print(f'(row, colum): {A_tilde.shape}, {B_tilde.shape}, {C_tilde.shape}, {D_tilde.shape}')

simplify(D_tilde)
s =
