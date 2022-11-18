# https://eeweb.engineering.nyu.edu/iselesni/ogs/

import scipy
import scipy.signal as spsig
import scipy.io.wavfile as spwav
import numpy as np
import sys


def check(x):
    try:
        assert np.all(np.isfinite(x))
        assert np.all(x > 0)
    except:
        print(x)

def lin2db(x):
    return 20*np.log10(x)

def gamma_inc_1(x):
    return 1 - np.exp(-x)

def gamma_inc_2(x):
    return 1 - (1 + x) * np.exp(-x)
    # return gamma_inc_1(x) - x * np.exp(-x)

def gain_srwf(xi, gamma):
    return np.sqrt(xi / (1 + xi))

def gain_wiener(xi, gamma):
    return xi / (1 + xi)

def gain_stsa(xi, gamma):
    # Y. Ephraim, D. Malah, Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator.
    nu = xi / (1 + xi) * gamma
    k = (1+nu)*scipy.special.iv(0, nu/2) + nu*scipy.special.iv(1, nu/2)
    return np.where(nu < 30, np.sqrt(np.pi) / 2 * np.sqrt(nu) / gamma * np.exp(-nu/2) * k, xi / (1 + xi))

def gain_lsa(xi, gamma):
    # Y. Ephraim, D. Malah, Speech enhancement using a minimum mean-square error log-spectral amplitude estimator.
    nu = xi / (1 + xi) * gamma
    return xi / (1 + xi) * np.exp(-scipy.special.expi(-nu) / 2)

def gain_lap(xi, gamma):
    # I. Cohen, Speech enhancement using super-Gaussian speech models and noncausal a priori SNR estimation
    L_p = 1/np.sqrt(xi) + np.sqrt(gamma)
    L_n = 1/np.sqrt(xi) - np.sqrt(gamma)
    k = 2/(L_p - L_n) * (L_p*scipy.special.erfcx(L_p) - L_n*scipy.special.erfcx(L_n)) / (scipy.special.erfcx(L_p) + scipy.special.erfcx(L_n))
    return np.where(np.isfinite(k), k, xi / (1 + xi))

def gain_implap(xi, gamma):
    # M. Rashidi-nejad et al, "Speech Enhancement using an Improved MMSE Estimator with Laplacian Prior"
    p = np.sqrt(1/xi) - np.sqrt(2*gamma)
    k = 1 / np.sqrt(2*gamma) * (1.32934/0.8862269) * scipy.special.pbdv(-2.5, p)[0] / scipy.special.pbdv(-1.5, p)[0]
    return np.where(np.isfinite(k), k, xi / (1 + xi))


class NoiseEstMMSELowCompl:
    '''
    Paper: Hendriks, R. C., Heusdens, R., & Jensen, J. (2010, March). MMSE based noise PSD tracking with low complexity. In 2010 IEEE International Conference on Acoustics, Speech and Signal Processing (pp. 4266-4269). IEEE.
    PDF: http://cas.et.tudelft.nl/pubs/0004266.pdf
    Implementation (NB bug on lines 37 and 39): https://github.com/steve3nto/NoiseReductionProject/blob/master/noise_estimation.m
    '''
    def __init__(self):
        self.var_nse = 1e-4
        self.xi_min = 1e-4
        self.alpha = 0.95
        self.beta = 0.8

    def estimate(self, Y_psd, X_psd=None):
        zeta = Y_psd / self.var_nse # A posteriori SNR
        xi_ml = np.maximum(zeta - 1, 0)
        var_nse_cond = (1/(1+xi_ml)**2 + xi_ml/(1+xi_ml)/zeta) * Y_psd
        # TODO: this is equivalent (could be faster)
        # var_nse_cond = Y_psd * 1/(1+xi_ml)**2 + self.var_nse * xi_ml/(1+xi_ml)

        # Estimate a-priori SNR via DD approach
        if X_psd is None:
            X_psd = np.maximum(Y_psd - self.var_nse, 0)
        xi_dd = np.maximum(self.alpha * (X_psd / var_nse_cond) + (1 - self.alpha) * xi_ml, self.xi_min)
        # TODO: is the line above effective? This seems to work almost the same way:
        # xi_dd = X_psd / var_nse_cond

        b = (1+xi_dd) * gamma_inc_2(1/(1+xi_dd)) + np.exp(-1 / (1+xi_dd))
        var_nse = var_nse_cond / b
        var_nse = spsig.convolve(var_nse, [0.1, 0.2, 0.4, 0.2, 0.1], mode='same')
        self.var_nse = self.beta * self.var_nse + (1 - self.beta) * var_nse
        return self.var_nse


class NoiseEstMMSEUnbiased:
    '''
    Paper: Gerkmann, T., & Hendriks, R. C. (2011). Unbiased MMSE-based noise power estimation with low complexity and low tracking delay. IEEE Transactions on Audio, Speech, and Language Processing, 20(4), 1383-1393.
    PDF: http://cas.et.tudelft.nl/pubs/gerkmann_unbiasedMMSE_TASL2012.pdf
    Improves on NoiseEstimator (Hendriks et al, 2010)
    '''
    def __init__(self):
        self.var_nse = 1e-5
        self.beta = 0.8
        p_h1 = 0.7
        xi_h1_db = 15 # dB SNR (power ratio)
        self.p_h0_over_h1 = (1 - p_h1) / p_h1
        self.xi_h1 = np.power(10, xi_h1_db / 10) 

    def estimate(self, Y_psd, X_psd=None):
        zeta = Y_psd / self.var_nse # A posteriori SNR

        p_h1_y = 1 / (1 + self.p_h0_over_h1 * (1 + self.xi_h1) * np.exp(-zeta * self.xi_h1 / (1 + self.xi_h1)))
        p_h1_y = np.minimum(p_h1_y, 0.99)

        var_nse = Y_psd * (1 - p_h1_y) + self.var_nse * p_h1_y # E(|N|^2 | y)

        # var_nse = spsig.convolve(var_nse, [0.1, 0.2, 0.4, 0.2, 0.1], mode='same')
        self.var_nse = self.beta * self.var_nse + (1 - self.beta) * var_nse
        return self.var_nse


class NoiseEstRegStat:
    '''
    Paper: Li, X., Girin, L., Gannot, S., & Horaud, R. (2016, March). Non-stationary noise power spectral density estimation based on regional statistics. In 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 181-185). IEEE.
    PDF: https://hal.inria.fr/hal-01250892/file/noise_psd.pdf
    Implementation: https://github.com/Audio-WestlakeU/RS_noisePSD
    '''
    def __init__(self):
        self.var_nse = 1e-5
        self.alpha_n = 0.8
        self.alpha_x = 0.85
        self.last_P = 0
        self.last_Y_psd = 0
        self.nu_nv = 0
        self.nu_ndv = 0
        self.nu_nav = 0
        self.nu_mcr = 0

    def estimate(self, Y_psd, X_psd=None):
        # First-order recursively smoothed periodogram
        P = self.last_P * self.alpha_x + Y_psd * (1 - self.alpha_x)

        # Normalized Variance
        self.nu_nv = self.alpha_x * self.nu_nv + (1 - self.alpha_x) * ((Y_psd - P) / P)**2
        # Normalized Differential Variance
        self.nu_ndv = self.alpha_x * self.nu_ndv + (1 - self.alpha_x) * ((Y_psd - self.last_Y_psd) / P)**2
        # Normalized Average Variance
        self.nu_nav = self.alpha_x * self.nu_nav + (1 - self.alpha_x) * (((Y_psd + self.last_Y_psd) / 2 - P) / P)**2
        # Median Crossing Rate
        self.nu_mcr = self.alpha_x * self.nu_mcr + (1 - self.alpha_x) * np.where((Y_psd - 0.69 * P) * (self.last_Y_psd - 0.69*self.last_P) < 0, 1, 0)

        self.last_P = P
        self.last_Y_psd = Y_psd

        return self.var_nse


def denoise(Y, gain=gain_lsa):
    N_bins = Y.shape[0]
    noise_est = NoiseEstMMSEUnbiased()
    alpha = 0.98
    xi_min = 0.01
    p_h1 = 0.8      # Probability of speech
    # p_h1 = np.array([0.8] * 100 + [0.4] * (N_bins - 100))
    G_min_db = -20  # -25 dB amplitude gain
    p_h0_over_h1 = (1 - p_h1) / p_h1
    G_min = np.power(10, G_min_db / 20)

    X = []
    W = []
    # X_psd = np.zeros(N_bins)
    xi_h1 = xi_min
    for Y_l in Y.transpose():
        Y_psd = np.abs(Y_l)**2
        lambda_nse = noise_est.estimate(Y_psd)  # Variance of noise
        zeta = Y_psd / lambda_nse       # A posteriori SNR

        # Estimate a-priori SNR via DD approach
        # xi = np.maximum(xi_min, alpha * X_psd / lambda_nse + (1 - alpha) * np.maximum(zeta - 1, 0))
        xi = np.maximum(xi_min, alpha * xi_h1 + (1 - alpha) * np.maximum(zeta - 1, 0))

        # Estimate speech probability
        p_h1_y = 1 / (1 + p_h0_over_h1 * (1 + xi) * np.exp(-zeta * xi / (1 + xi)))
        p_h1_y = spsig.convolve(p_h1_y, [0.2, 0.6, 0.2], mode='same')
        # p_h1_y = 1

        # Calculate gain
        G_H1 = gain(xi, zeta)
        G = np.power(G_H1, p_h1_y) * np.power(G_min, 1 - p_h1_y)
        # G = spsig.convolve(G, [0.2, 0.6, 0.2], mode='same')
        X_l = G * Y_l

        xi_h1 = G_H1**2 * zeta

        X.append(X_l)
        W.append(np.sqrt(lambda_nse))
        # W.append(p_h1_y)
    return np.array(X).transpose(), np.array(W).transpose()


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--play", action="store_true")
parser.add_argument("--save")
args = parser.parse_args()

fs, sig = spwav.read(args.input_file)
sig = sig / 32768.0 + 1e-6
# sig = np.ones(len(sig))
print(f'Read {len(sig)} samples at {fs} Hz rate (min/max={np.min(sig)}/{np.max(sig)})', file=sys.stderr)

win_size = int(0.032 * fs + 0.5)
# win_size = 512
win_shift = win_size // 2
nfft = max(win_size, 512)
print(f'Using window size of {win_size} samples')
win = spsig.get_window('hamming', win_size)
_, _, H = spsig.stft(sig, fs=fs, window=win, nperseg=win_size, noverlap=(win_size - win_shift), nfft=nfft)
print(f'STFT size: {H.shape} min/max={lin2db(np.min(np.abs(H)))}/{lin2db(np.max(np.abs(H)))} dB')

H2, W = denoise(H)

_, sig2 = spsig.istft(H2, fs=fs, window=win, nperseg=win_size, noverlap=(win_size - win_shift), nfft=nfft)

if args.save:
    print(f'Writing {len(sig2)} samples')
    spwav.write(args.save, fs, (0.5 + np.clip(sig2, -1, 1) * 32767).astype(np.int16))
    
if args.play:
    import sounddevice as sd
    # stream = sd.OutputStream(samplerate=fs, channels=1)
    sd.play(sig2, fs)
    status = sd.wait()

if args.plot:
    import matplotlib.pyplot as plt

    plt.figure()
    # x_db = np.linspace(-20, 20, 100)
    # zeta = 1 + np.power(10, x_db/10)
    # for xi_db in [20, 15, 5, 0, -5, -10, -15, -20]:
    #     xi = np.power(10, xi_db / 10)
    #     y1 = gain_lsa(xi, zeta)
    #     plt.plot(x_db, lin2db(y1))
    #     y2 = gain_stsa(xi, zeta)
    #     plt.plot(x_db, lin2db(y2), '--')
    # plt.ylim([-35, 5])
    x_db = np.linspace(-20, 20, 100)
    xi = np.power(10, x_db/10)
    for zeta_db in [-15, -7, 0, 7, 15]:
        zeta = np.power(10, zeta_db / 10)
        y = gain_lap(xi, zeta)
        plt.plot(x_db, lin2db(y))
    plt.grid()

    plt.figure()
    plt.subplot(311)
    plt.imshow(lin2db(np.abs(H)), origin='lower', vmin=-70, vmax=-40) # cmap='gray_r', 
    plt.subplot(312)
    # plt.imshow(lin2db(H2), origin='lower')
    plt.imshow(lin2db(np.abs(H2)), origin='lower', vmin=-70, vmax=-40)
    # plt.imshow(lin2db(np.abs(H2)), origin='lower', cmap='gray_r', vmin=-70, vmax=-20)
    plt.subplot(313)
    plt.imshow(lin2db(W), origin='lower', vmin=-70, vmax=-40)
    # plt.imshow(lin2db(W) / 2, origin='lower', vmin=-10, vmax=10)
    # plt.plot(np.mean(lin2db(W) / 2, axis=0))
    # plt.plot(np.mean(W[6:100, :], axis=0))

    # plt.figure()
    # plt.plot(lin2db(np.abs(H[:, 610])))
    # plt.plot(lin2db(np.abs(H2[:, 610])))
    # plt.plot(lin2db(np.abs(W[:, 610])))

    plt.show()
