import os
import wave

import fluidsynth
import numpy as np
from midi2audio import FluidSynth

if __name__ == "__main__":


    fs = FluidSynth(os.path.join(os.getcwd()+'resources/soundfiles/_GD__Clean_Grand_Mistral/Clean_Grand_Mistral.sf2'))
    #fs.play_midi(os.path.join(os.getcwd()+'resources/midi/new_song.mid'))
    fs.midi_to_audio('C:\\Users\\fabia\\PycharmProjects\\MidiConnection\\resources\midi\\new_song.mid', 'output.wav')

    #FluidSynth('sound_font.sf2').midi_to_audio('input.mid', 'output.wav')