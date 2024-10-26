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


def synthesize_fluidsynth(synth, notes, channel=0, sample_rate=44100, plot=False):
    event_list = []
    for note in notes:
        event_list += [[note.start, 'note on', note.pitch, note.velocity]]
        event_list += [[note.end, 'note off', note.pitch]]
    # Sort the event list by time, and secondarily by whether the event
    # is a note off
    event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
    # Add some silence at the beginning according to the time of the first
    # event
    current_time = event_list[0][0]
    # Convert absolute seconds to relative samples
    next_event_times = [e[0] for e in event_list[1:]]
    for event, end in zip(event_list[:-1], next_event_times):
        event[0] = end - event[0]
    # Set silence duration at the end to zero
    event_list[-1][0] = 0.

    # Pre-allocate output array
    total_time = current_time + np.sum([e[0] for e in event_list])
    synthesized = np.zeros(int(np.ceil(sample_rate * total_time)))
    # Iterate over all events
    for event in event_list:
        # Process events based on type
        if event[1] == 'note on':
            synth.noteon(channel, event[2], event[3])
        elif event[1] == 'note off':
            synth.noteoff(channel, event[2])
        elif event[1] == 'pitch bend':
            synth.pitch_bend(channel, event[2])
        elif event[1] == 'control change':
            synth.cc(channel, event[2], event[3])
        # Add in these samples
        current_sample = int(sample_rate * current_time)
        end = int(sample_rate * (current_time + event[0]))
        samples = synth.get_samples(end - current_sample)[::2]
        synthesized[current_sample:end] += samples
        # Increment the current sample
        current_time += event[0]
    # TODO Normalize or not ?
    #synthesized /= np.abs(synthesized).max()

    if plot:
        librosa.display.waveshow(wav_data, sample_rate=sample_rate)
        plt.show()
        cqt_data = librosa.cqt(wav_data, hop_length=512, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(np.abs(cqt_data), ref=np.max),
                                 sr=sample_rate, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q power spectrum')
        plt.show()

    return synthesized


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