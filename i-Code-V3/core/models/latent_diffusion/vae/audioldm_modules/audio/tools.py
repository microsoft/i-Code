import torch
import numpy as np
import torchaudio


def get_mel_from_wav(audio, _stft):
    dtype = audio.dtype
    audio = torch.clip(audio.unsqueeze(0), -1, 1).cuda()
    audio = torch.autograd.Variable(audio, requires_grad=False).to(dtype)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0)
    log_magnitudes_stft = (
        torch.squeeze(log_magnitudes_stft, 0)
    )
    energy = torch.squeeze(energy, 0)
    return melspec, log_magnitudes_stft, energy


def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank


def pad_wav(waveform, segment_length):
    batch_size, waveform_length = waveform.shape
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:, :segment_length]
    elif waveform_length < segment_length:
        temp_wav = torch.zeros((batch_size, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav


def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform, 1, keepdim=True)
    waveform = waveform / (torch.max(torch.abs(waveform), 1).values + 1e-8).unsqueeze(1)
    return waveform * 0.5


def process_wav_file(waveform, segment_length):
    waveform = normalize_wav(waveform)
    waveform = pad_wav(waveform, segment_length)
    return waveform


def wav_to_fbank(waveform, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    # mixup
    waveform = process_wav_file(waveform, target_length * 160)  # hop size is 160
    outputs = []
    for waveform_i in waveform:
        fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform_i, fn_STFT)
        fbank = fbank[:, :target_length].T
        outputs.append(fbank)
    return torch.stack(outputs, 0).unsqueeze(1)