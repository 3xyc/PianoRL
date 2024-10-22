import struct

import librosa
import musicpy.musicpy
import numpy as np
from matplotlib import pyplot as plt
from musicpy.daw import daw
from musicpy.structures import chord, note

new_song = daw(1, name='piano song')
new_song.load(0, r'resources/soundfiles/[GD] Clean Grand Mistral.sf2')
# export piece object to wav
wav = new_song.export(note('C', 5), mode='wav', action='get', show_msg=True)
# export piece object to midi stream
midi_stream = musicpy.musicpy.write(note('C', 5),
                                    bpm=120,
                                    channel=0,
                                    start_time=None,
                                    save_as_file=False)
print(midi_stream.getvalue())

midi_object = musicpy.musicpy.read(midi_stream, is_file=True)
print(midi_object)
print(wav)
print(wav[0:1000])
raw_data = wav.raw_data
# Convert the raw data to integers (16-bit PCM)
# 'h' format is used for signed 16-bit (2 bytes)
num_samples = len(raw_data) // 2  # Each sample is 2 bytes (16-bit PCM)
audio_samples = struct.unpack('<' + 'h' * num_samples, raw_data)  # Little-endian format

# Convert to numpy array for easier manipulation
audio_samples_np = np.array(audio_samples, dtype=np.float32)

# Apply librosa's CQT or other processing directly (assuming 44100Hz sample rate)
sr = 44100  # Sample rate (you should adjust this based on your audio file)
cqt = librosa.cqt(audio_samples_np, sr=sr)
print(cqt)
# Display the CQT spectrogram
librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt), ref=np.max),
                         sr=sr, x_axis='time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.show()
print(audio_samples_np)
print(wav)