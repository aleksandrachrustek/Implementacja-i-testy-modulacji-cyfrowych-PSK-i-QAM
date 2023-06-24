import numpy as np
import matplotlib.pyplot as plt

#define the carrier frequency and symbol period
fc = 1000  #carrier frequency (Hz)
Tsym = 0.01 #symbol period (seconds)

#define the constellation points
constellation_4QAM = [1+1j, -1+1j, -1-1j, 1-1j]
constellation_16QAM = [3+3j, 1+3j, -1+3j, -3+3j, 3+1j, 1+1j, -1+1j, -3+1j, 3-1j, 1-1j, -1-1j, -3-1j, 3-3j, 1-3j, -1-3j, -3-3j]

#define the data to be transmitted
data_4QAM = [0, 1, 2, 3, 0, 1, 2, 3]
data_16QAM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

#choose 4-QAM or 16-QAM
constellation = constellation_4QAM
data = data_4QAM

#time vector
Fs = len(data)/Tsym
t = np.linspace(0, len(data)*Tsym, int(Fs*len(data)))

#create the modulated signal
sig = np.zeros(len(t), dtype=np.complex64)
for i, sym in enumerate(data):
    sig[i*int(Fs*Tsym):(i+1)*int(Fs*Tsym)] = constellation[sym]*np.exp(2j*np.pi*fc*t[i*int(Fs*Tsym):(i+1)*int(Fs*Tsym)])

#add noise
SNRdB =13 #signal to noise ratio (in dB)
signal_power = np.sum(np.abs(sig)**2)/len(sig)
noise_power = signal_power/(10**(SNRdB/10))
noise = np.sqrt(noise_power/2)*(np.random.randn(len(sig))+1j*np.random.randn(len(sig)))
noisy_sig = sig + noise

#plot the modulated signal and noisy signal
fig, axs = plt.subplots(2)
fig.suptitle('QAM Modulated Signal with Noise')
axs[0].plot(t, sig.real, label='I')
axs[0].plot(t, sig.imag, label='Q')
axs[0].set_xlabel('Time(seconds)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Modulated Signal')
axs[0].legend()
axs[1].plot(t, noisy_sig.real, label='I')
axs[1].plot(t, noisy_sig.imag, label='Q')
axs[1].set_xlabel('Time(seconds)')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Noisy Signal')
axs[1].legend()
plt.show()

# Perform QAM demodulation
demodulated_data = []
for i in range(len(data)):
    symbol = constellation[data[i]]
    segment = noisy_sig[i*int(Fs*Tsym):(i+1)*int(Fs*Tsym)]
    
    # Perform correlation
    correlation = np.conj(symbol) * segment
    
    # Find the index of the closest constellation point
    closest_index = np.argmin(np.abs(correlation - symbol))
    
    # Store the demodulated data
    demodulated_data.append(closest_index)

# Calculate the error bit rate
error_count = sum(bit1 != bit2 for bit1, bit2 in zip(data, demodulated_data))
error_bit_rate = error_count / len(data)

# Print the error bit rate
print('Error bit rate:', error_bit_rate)

# Plot the demodulated data
plt.stem(demodulated_data, linefmt='C3-', markerfmt='C3o', basefmt='C0-')
plt.xlabel('Symbol Index')
plt.ylabel('Demodulated Data')
plt.title('Demodulated Data')
plt.show()
