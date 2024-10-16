import time
import fluidsynth
"""
soundfile_path = "/resources/soundfiles/[GD] Clean Grand Mistral.sf2"
midi_path = "C:\\Users\\fabia\\PycharmProjects\\MidiConnection\\resources\\midi\\resources\\midi\\new_song.mid"
output_path = "C:\\Users\\fabia\\PycharmProjects\\MidiConnection\\resources\\wav\\new_song.wav"
sample_rate = 44100


#the follwoing is the example provided at https://www.fluidsynth.org/api/FileRenderer.html,
# adapted to utilize the python wrappers of the c functions.
settings = fluidsynth.new_fluid_settings()

# specify the file to store the audio to
# make sure you compiled fluidsynth with libsndfile to get a real wave file
# otherwise this file will only contain raw s16 stereo PCM


fluidsynth.fluid_settings_setstr(settings, "audio.file.name".encode(), "e".encode())

# use number of samples processed as timing source, rather than the system timer
fluidsynth.fluid_settings_setstr(settings, "player.timing-source".encode(), "sample".encode())

# since this is a non-realtime scenario, there is no need to pin the sample data
fluidsynth.fluid_settings_setint(settings, "synth.lock-memory".encode(), 0)

synth = fluidsynth.Synth(settings)
# Load the SoundFont
sfid = synth.sfload(soundfile_path)

selfsynth = synth.synth

player = fluidsynth.new_fluid_player(selfsynth)
fluidsynth.fluid_player_add(player, "resources/midi/new_song.mid".encode())
fluidsynth.fluid_player_play(player)

renderer = fluidsynth.new_fluid_file_renderer(selfsynth)

while (fluidsynth.fluid_player_get_status(player) == fluidsynth.FLUID_PLAYER_PLAYING):
    if (fluidsynth.fluid_file_renderer_process_block(renderer) != fluidsynth.FLUID_OK):
        print("NOT OKAY")
        break


fluidsynth.fluid_player_stop(player)
fluidsynth.fluid_player_join(player)

fluidsynth.delete_fluid_file_renderer(renderer)
fluidsynth.delete_fluid_player(player)
fluidsynth.delete_fluid_synth(selfsynth)
fluidsynth.delete_fluid_settings(settings)

"""