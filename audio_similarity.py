import librosa
import numpy as np
import fluidsynth
from matplotlib import pyplot as plt
from pretty_midi import pretty_midi
from midi_util import synthesize_fluidsynth
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

from pesq import pesq
def persq_similarity(wav_1, wav_2, sample_rate=44100):
    """Perceptual Evaluation of Audio Quality (PEAQ): PEAQ is a standard for assessing the quality of audio by modeling human auditory perception. It's used for objective evaluation of lossy compression algorithms. While implementations are more complex, there are Python libraries (like pesq for speech quality) that approximate it.

    Pros: Closely aligned with human perception of quality.
    Cons: More complex to implement, designed primarily for speech and simple signals.

PESQ (Perceptual Evaluation of Speech Quality): A method for objectively assessing the quality of speech, often used for telecommunications. It can give you an idea of how perceptually similar two signals are."""
    pesq_score = pesq(sample_rate, wav_1, wav_2, 'wb')
    return pesq_score

def frequency_domain_similarity(wav_1, wav_2, sample_rate=44100):
    # Compute the CQT for both original and synthesized signals
    original_cqt = librosa.cqt(wav_1, sr=sample_rate)
    synthesized_cqt = librosa.cqt(wav_2, sr=sample_rate)

    # Flatten and compute cosine similarity
    original_cqt_flat = np.abs(original_cqt).flatten()
    synthesized_cqt_flat = np.abs(synthesized_cqt).flatten()
    return cosine_similarity([original_cqt_flat], [synthesized_cqt_flat])
def mel_frequency_cepstral_coefficients(wav_1, wav_2, sample_rate=44100):

    original_mfcc = librosa.feature.mfcc(y=wav_1, sr=sample_rate, n_mfcc=13)
    synthesized_mfcc = librosa.feature.mfcc(y=wav_2, sr=sample_rate, n_mfcc=13)

    # Compute Euclidean distance or cosine similarity between MFCCs
    mfcc_similarity = cosine_similarity(original_mfcc.T, synthesized_mfcc.T)
    avg_similarity = np.mean(mfcc_similarity)
    return avg_similarity
def structural_similarity_index_ssim(wav_1, wav_2):
    original_spectrogram = np.abs(librosa.stft(wav_1))
    synthesized_spectrogram = np.abs(librosa.stft(wav_2))

    # Normalize spectrograms to [0, 1] for compatibility with SSIM
    original_spectrogram /= np.max(original_spectrogram)
    synthesized_spectrogram /= np.max(synthesized_spectrogram)

    # Compute SSIM over the spectrograms
    ssim_value, _ = ssim(original_spectrogram, synthesized_spectrogram, data_range=1.0, full=True)
    return ssim_value
#def cosine_similarity(wav_1, wav_2):

if __name__ == "__main__":
    sound_file = 'resources/soundfiles/[GD] Clean Grand Mistral.sf2'
    print("s")
    synth = fluidsynth.Synth()
    sfid = synth.sfload(sound_file)
    synth.program_select(0, sfid, 0, 0)

    n1 = pretty_midi.Note(60, 80, 0,5)
    n2 = pretty_midi.Note(60, 7, 0,5)

    notes_0 = [n1, n2]
    notes_1 = [n1]
    notes_2 = [n2]
    print("s")
    wav_0 = synthesize_fluidsynth(synth, notes_0)
    cqt_data = librosa.cqt(wav_0)

    librosa.display.specshow(librosa.power_to_db(np.abs(cqt_data), ref=np.max),
                             sr=44100, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()

    wav_1 = synthesize_fluidsynth(synth, notes_1)
    cqt_data = librosa.cqt(wav_1)

    librosa.display.specshow(librosa.power_to_db(np.abs(cqt_data), ref=np.max),
                             sr=44100, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()

    wav_2 = synthesize_fluidsynth(synth, notes_2)
    cqt_data = librosa.cqt(wav_2)

    librosa.display.specshow(librosa.power_to_db(np.abs(cqt_data), ref=np.max),
                             sr=44100, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()

    ssimr = structural_similarity_index_ssim(wav_1, wav_2)
    fr = frequency_domain_similarity(wav_1, wav_2)
    flcc = mel_frequency_cepstral_coefficients(wav_1, wav_2),

    print("1-2", ssimr, fr, flcc)

    ssimr = structural_similarity_index_ssim(wav_0, wav_2)
    fr = frequency_domain_similarity(wav_0, wav_2)
    flcc = mel_frequency_cepstral_coefficients(wav_0, wav_2),

    print("0-2", ssimr, fr, flcc)

    ssimr = structural_similarity_index_ssim(wav_1, wav_0)
    fr = frequency_domain_similarity(wav_1, wav_0)
    mfcc = mel_frequency_cepstral_coefficients(wav_1, wav_0),

    print("1-0", ssimr, fr, mfcc)


    print(wav_0.shape)
    print(wav_1.shape)
    print(wav_2.shape)

    print(wav_1+wav_2)
    print(wav_0)
    print(wav_1)
    print(wav_2)

    librosa.display.waveshow(wav_0, sr=44100)
    plt.show()
    librosa.display.waveshow(wav_1, sr=44100)
    plt.show()
    librosa.display.waveshow(wav_2, sr=44100)
    plt.show()
    librosa.display.waveshow(wav_1+wav_2, sr=44100)
    plt.show()
    print(mel_frequency_cepstral_coefficients(wav_0, wav_1+wav_2))

    cqt_data = librosa.cqt(wav_1+wav_2)
    librosa.display.specshow(librosa.power_to_db(np.abs(cqt_data), ref=np.max),
                             sr=44100, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Constant-Q power spectrum')
    plt.show()

