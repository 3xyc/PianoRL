import time
import rtmidi

def send_midi():
    midiout = rtmidi.MidiOut()

    available_ports = midiout.get_ports()

    midiout.open_port(1)  # Open correct port


    with midiout:
        # Example of sending a Note On and Note Off message
        note_on = [0x90, 60, 112]  # channel 1, middle C, velocity 112
        note_off = [0x80, 60, 0]

        midiout.send_message(note_on)
        time.sleep(0.5)
        midiout.send_message(note_off)
        time.sleep(0.1)


def midi_callback(message, data):
    # message is a tuple (list of MIDI bytes, timestamp)
    midi_bytes, timestamp = message
    print(f"Received MIDI message: {midi_bytes} at time {timestamp}")


if __name__ == "__main__":

    # Receiving MIDI
    midiin = rtmidi.MidiIn()
    available_ports = midiin.get_ports()
    print("Available Input Ports:", available_ports)

    midiin.open_port(0)  # Open correct input port


    # Set callback for receiving MIDI messages
    midiin.set_callback(midi_callback)

    print("Listening for MIDI messages...")

    try:
        while True:
            time.sleep(0.1)  # Keep the script running to listen for messages
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        midiin.close_port()
