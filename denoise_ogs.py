# https://eeweb.engineering.nyu.edu/iselesni/ogs/

import scipy.signal as spsig
import scipy.io.wavfile as spwav
import numpy as np
import sys


def lin2db(x):
    return 20*np.log10(x)

K_LAMBDA = {
    (1, 1): 3.86,
    (1, 2): 2.00,
    (1, 3): 1.35,
    (1, 4): 1.04,
    (1, 5): 0.86,
    (2, 2): 1.01,
    (2, 3): 0.69,
    (2, 4): 0.55,
    (2, 5): 0.47,
    (2, 8): 0.36,
    (3, 3): 0.50,
    (3, 4): 0.42,
    (3, 5): 0.37,
    (3, 6): 0.34,
    (4, 4): 0.36,
    (4, 5): 0.32,
    (5, 5): 0.28
}

def denoise(Y, sigma, K1=8, K2=2, n_iter=25):
    k_lambda = K_LAMBDA[(min(K1, K2), max(K1, K2))]
    A = Y
    lbd = k_lambda * sigma
    kernel = np.ones((K1, K2))
    for iter in range(n_iter):
        R = np.sqrt(spsig.convolve2d(np.abs(A)**2, kernel, mode='full'))
        V = 1 / (1 + spsig.convolve2d(1/R, kernel, mode='full') * lbd[:, np.newaxis]) 
        V = V[K1-1:-K1+1, K2-1:-K2+1]
        print(f'Iteration {iter} max(V)={np.max(V)} min(V)={np.min(V)}')
        A = Y * V
    return A

fs, sig = spwav.read(sys.argv[1])
sig = sig / 32768.0 + 1e-6
# sig = np.ones(len(sig))
print(f'Read {len(sig)} samples at {fs} Hz rate (min/max={np.min(sig)}/{np.max(sig)})')

win_size = int(0.032 * fs + 0.5)
# win_size = 512
win_shift = win_size // 2
nfft = win_size
print(f'Using window size of {win_size} samples')
win = spsig.get_window('hamming', nfft)
_, _, H = spsig.stft(sig, fs=fs, window=win, nperseg=win_size, noverlap=(win_size - win_shift), nfft=nfft)
print(f'STFT size: {H.shape} min/max={lin2db(np.min(np.abs(H)))}/{lin2db(np.max(np.abs(H)))} dB')

# sigma = 0.0015
sigma_vec = np.linspace(0.0015, 0.0030, 140)
H2 = denoise(H, sigma_vec)

# H2 = H2 / 20
_, sig2 = spsig.istft(H2, fs=fs, window=win, nperseg=win_size, noverlap=(win_size - win_shift), nfft=nfft)

# H2 = H * np.abs(H2)**2 / (np.abs(H2)**2 + sigma*sigma)

print(f'Writing {len(sig2)} samples')
spwav.write(sys.argv[2], fs, sig2)

import matplotlib.pyplot as plt

plt.subplot(211)
plt.imshow(lin2db(np.abs(H[:, 400:800])), origin='lower', cmap='gray_r', vmin=-70, vmax=-20)
plt.subplot(212)
plt.imshow(lin2db(np.abs(H2[:, 400:800])), origin='lower', cmap='gray_r', vmin=-70, vmax=-20)
plt.show()
