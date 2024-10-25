import os
import subprocess
import tempfile
import time

import librosa
import mido
from io import BytesIO

import numpy as np
import pretty_midi
from matplotlib import pyplot as plt


def play_mido(midi):
    output_names = mido.get_output_names()
    with mido.open_output(output_names[0]) as port:
        for message in midi:
            print(message)
            if message.is_meta:
                continue
            time.sleep(message.time)
            port.send(message)



def fast_fluidsynth(m, sr=44100):
    '''
    Faster fluidsynth synthesis using the command-line program
    instead of pyfluidsynth.
    Parameters
    ----------
    - m : pretty_midi.PrettyMIDI
        Pretty MIDI object
    - fs : int
        Sampling rate
    Returns
    -------
    - midi_audio : np.ndarray
        Synthesized audio, sampled at fs
    '''
    # Write out temp mid file
    midi_stream = BytesIO()
    midi_temp_name="temp.mid"
    m.save(file=midi_stream)
    print(midi_stream.getvalue())
    # Get path to temporary .wav file
    print(1)
    wav_temp_name = "temp.wav"
    # Get path to default pretty_midi SF2
    sf2_path = "resources/soundfiles/[GD] Clean Grand Mistral.sf2"
    # Make system call to fluidsynth
    with open(os.devnull, 'w') as devnull:
        print(2)
        subprocess.check_output(
            ['fluidsynth', '-F', wav_temp_name, '-r', str(sr), sf2_path,
             midi_temp_name], stderr=devnull)
    # Load from temp wav file
    audio, _ = librosa.load(wav_temp_name, sr=sr)
    # Occasionally, fluidsynth pads a lot of silence on the end, so here we
    # crop to the length of the midi object
    audio = audio[:int(m.length * sr)]
    print(audio)
    return audio

def pretty_midi_to_cqt_and_plot(mid, sample_rate=44100, hop_length=512):
    wav_data = mid.fluidsynth(sample_rate, sf2_path="resources/soundfiles/[GD] Clean Grand Mistral.sf2")
    cqt_data = librosa.cqt(wav_data, hop_length=hop_length, sr=sample_rate)

    librosa.display.specshow(librosa.power_to_db(np.abs(cqt_data), ref=np.max),
                             sr=sample_rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()



if __name__ == '__main__':
    midi = mido.MidiFile("new_song.mid")
    #wav = fast_fluidsynth(midi)

    p_m = pretty_midi.PrettyMIDI("new_song.mid")
    print(p_m.get_pitch_class_histogram())
    p_m = pretty_midi.PrettyMIDI("piano.mid")
    print(p_m.get_pitch_class_histogram())
    exit()
    sr = 44100
    wav_data = p_m.fluidsynth(sr, "resources/soundfiles/[GD] Clean Grand Mistral.sf2")

    print(wav_data)

    cqt_data = librosa.cqt(wav_data,  sr=sr)

    print(wav_data.shape)
    print(cqt_data.shape)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(cqt_data), ref=np.max),
                             sr=sr, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()