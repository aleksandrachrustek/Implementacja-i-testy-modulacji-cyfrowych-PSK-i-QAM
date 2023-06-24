import numpy as np
import matplotlib.pyplot as plt

#define the carrier frequency and bit rate
fc = 2402e6 #bluetooth carrier frequency
bit_rate = 1e6 #bluetooth bit rate

#define the message to be transmitted
message = '1010101010101010'

#convert the message to binary
binary_message = ''
for char in message:
    binary_message += '{0:08b}'.format(ord(char))

#define the frequency deviation for each bit
delta_f = 1/8* bit_rate

#create the time base of the signal
time_step = 1/(1 * fc)
time = np.arange(0, len(binary_message) * 1/bit_rate, time_step)

#create the carrier signal
carrier = np.cos(2 * np.pi * fc * time)

#create the modulated signal
modulated_signal = np.zeros_like(carrier)
for i in range(len(binary_message)):
    if binary_message[i] == '0':
        modulated_signal[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)] = np.cos(2 * np.pi * (fc - delta_f) * time[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)])
    else:
        modulated_signal[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)] = np.cos(2 * np.pi * (fc + delta_f) * time[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)])

#plot the modulated signal
plt.plot(time, modulated_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Modulated Signal (without interference)')
plt.show()

#add interference
interference = np.sin(2 * np.pi * 2* delta_f * time)
modulated_signal_with_interference = modulated_signal + interference

#plot the signal with interference
plt.plot(time, modulated_signal_with_interference)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Modulated Signal (with interference)')
plt.show()

# Perform coherent demodulation
demodulated_signal = np.zeros_like(modulated_signal_with_interference)

for i in range(len(binary_message)):
    if binary_message[i] == '0':
        demodulated_signal[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)] = modulated_signal_with_interference[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)] * np.cos(2 * np.pi * (fc - delta_f) * time[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)])
    else:
        demodulated_signal[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)] = modulated_signal_with_interference[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)] * np.cos(2 * np.pi * (fc + delta_f) * time[i*int(1/bit_rate/time_step):(i+1)*int(1/bit_rate/time_step)])

# Perform low-pass filtering
from scipy import signal

# Define the cutoff frequency for the low-pass filter
cutoff_freq = bit_rate / 2

# Create the low-pass filter
b, a = signal.butter(5, cutoff_freq, fs=1/time_step)

# Apply the low-pass filter to the demodulated signal
filtered_signal = signal.lfilter(b, a, demodulated_signal)

# Calculate the received message
received_message = ''
for i in range(len(binary_message)):
    if filtered_signal[i*int(1/bit_rate/time_step)] > 0:
        received_message += '1'
    else:
        received_message += '0'

# Calculate the error bit rate
error_count = sum(bit1 != bit2 for bit1, bit2 in zip(binary_message, received_message))
error_bit_rate = error_count / len(binary_message)

# Print the error bit rate
print('Error bit rate:', error_bit_rate)

# Plot the demodulated and filtered signal
plt.plot(time, filtered_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Demodulated and Filtered Signal')
plt.show()
