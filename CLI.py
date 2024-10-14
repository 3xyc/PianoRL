import subprocess

return_code = subprocess.call("echo Hello World", shell=True)
print(return_code)

command = "fluidsynth -ni soundfiles\[GD] Clean Grand Mistral.sf2 midi\new_song.mid -F output.wav -r 44100"
commands = ["fluidsynth", "-ni", "resources\\soundfiles\\[GD] Clean Grand Mistral.sf2", "resources\\midi\\new_song.mid",
           "-F", "output.wav", "-r", "44100"]
subprocess.run(commands, shell=True)