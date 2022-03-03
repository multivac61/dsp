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


# %%
# Bode plot with mix between low-band-highpass filters with 40db/dec

def dry_wet(mix):
    assert 0.0 <= mix <= 1.0, f'Expected mix in [0, 1], got {mix:.2f}'
    return 2 * np.array(
        [0.5 - mix, mix, 0] if 0.0 <= mix <= 0.5 else
        [0, 1 - mix, mix - 0.5]
    )


hs = [lowpass(wc, k, r), bandpass(wc, k, r), highpass(wc, k, r)]


def bode(mix, ws=np.linspace(0, 2 * np.pi * 44.1e3, 10_000)):
    h_den = sum(y.den for y in hs)

    padding = max(map(len, (y.num for y in hs)))
    h_num = sum(k * np.pad(y, (padding - len(y), 0)) for k, y in zip(dry_wet(mix), (y.num for y in hs)))

    return signal.bode((h_num, h_den), ws)


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

# %%
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
