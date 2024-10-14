import wave

import fluidsynth
import numpy as np

if __name__ == "__main__":
    soundfile_path = "resources/soundfiles/_GD__Clean_Grand_Mistral/[GD] Clean Grand Mistral.sf2"
    midi_path = "resources/midi/new_song.mid"
    output_path = "resources/wav/new_song.wav"
    sample_rate = 44100

    # Initialize FluidSynth
    fs = fluidsynth.Synth(samplerate=sample_rate, dr)
    # Load the SoundFont
    sfid = fs.sfload(soundfile_path)

    # Default to the first preset (can be customized)
    fs.program_select(0, sfid, 0, 0)


    fs.play_midi_file(midi_path)
    settings = fluidsynth.new_fluid_settings()
    driver = fluidsynth.new_fluid_audio_driver(settings, sfid)

    # Render the MIDI file into an audio buffer
    fs.start(driver=driver)  # Use any dummy driver that doesn't require a real device

    # Clean up FluidSynth instance
    fs.delete()
