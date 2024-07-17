"""Copyright: Nabarun Goswami (2024)."""
import os
import torch
import torch.nn as nn
import numpy as np
from scipy.io.wavfile import write
import torchaudio
from transformers.modeling_utils import ModuleUtilsMixin

from hierspeechpp import utils
from hierspeechpp.Mels_preprocess import MelSpectrogramFixed
from torch.nn import functional as F
from hierspeechpp.hierspeechpp_speechsynthesizer import (
    SynthesizerTrn, Wav2vec2
)
from hierspeechpp.ttv_v1.text import text_to_sequence
from hierspeechpp.ttv_v1.t2w2v_transformer import SynthesizerTrn as Text2W2V
from hierspeechpp.speechsr24k.speechsr import SynthesizerTrn as SpeechSR24
from hierspeechpp.speechsr48k.speechsr import SynthesizerTrn as SpeechSR48
from hierspeechpp.denoiser.generator import MPNet
from hierspeechpp.denoiser.infer import denoise

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

from huggingface_hub import snapshot_download


def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def add_blank_token(text):
    text_norm = intersperse(text, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def get_yaapt_f0(audio, rate=16000, interp=False):
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2

    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0, 'f0_max': 1100})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]
    f0 = np.vstack(f0s)
    return f0


class HierspeechppInferenceModel(nn.Module, ModuleUtilsMixin):
    def __init__(self,
                 hf_repo="subatomicseer/hierspeechpp_checkpoints",
                 output_sr=16000,
                 scale_norm='max',
                 denoise_ratio: float = 0.8,
                 noise_scale_vc: float = 0.333,
                 noise_scale_ttv: float = 0.333,
                 ):
        super().__init__()

        ckpts_dir = snapshot_download(hf_repo)

        ckpt_hierspeechpp = os.path.join(ckpts_dir, 'hierspeechpp_eng_kor', 'hierspeechpp_v1.1_ckpt.pth')
        ckpt_text2w2v = os.path.join(ckpts_dir, 'ttv_libritts_v1', 'ttv_lt960_ckpt.pth')
        ckpt_denoiser = os.path.join(ckpts_dir, 'denoiser', 'g_best')
        ckpt_sr = os.path.join(ckpts_dir, 'speechsr24k', 'G_340000.pth')
        ckpt_sr48 = os.path.join(ckpts_dir, 'speechsr48k', 'G_100000.pth')

        self.output_sr = output_sr
        self.scale_norm = scale_norm
        self.denoise_ratio = denoise_ratio
        self.noise_scale_vc = noise_scale_vc
        self.noise_scale_ttv = noise_scale_ttv

        hps = utils.get_hparams_from_file(os.path.join(os.path.split(ckpt_hierspeechpp)[0], 'config.json'))
        hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(ckpt_denoiser)[0], 'config.json'))
        hps_t2w2v = utils.get_hparams_from_file(os.path.join(os.path.split(ckpt_text2w2v)[0], 'config.json'))

        self.mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate,
            n_fft=hps.data.filter_length,
            win_length=hps.data.win_length,
            hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin,
            f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels,
            window_fn=torch.hann_window
        )

        self.text2w2v = Text2W2V(hps.data.filter_length // 2 + 1,
                                 hps.train.segment_size // hps.data.hop_length,
                                 **hps_t2w2v.model)
        self.text2w2v.load_state_dict(torch.load(ckpt_text2w2v))
        self.text2w2v.eval()

        self.w2v = Wav2vec2()

        self.net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                                    hps.train.segment_size // hps.data.hop_length,
                                    **hps.model)

        self.net_g.load_state_dict(torch.load(ckpt_hierspeechpp))
        self.net_g.eval()

        if output_sr == 48000:
            h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(ckpt_sr48)[0], 'config.json'))
            self.speechsr = SpeechSR48(h_sr48.data.n_mel_channels,
                                       h_sr48.train.segment_size // h_sr48.data.hop_length,
                                       **h_sr48.model)
            utils.load_checkpoint(ckpt_sr48, self.speechsr, None)
            self.speechsr.eval()
            self.hps_sr48 = h_sr48
        elif output_sr == 24000:
            h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(ckpt_sr)[0], 'config.json'))
            self.speechsr = SpeechSR24(h_sr.data.n_mel_channels,
                                       h_sr.train.segment_size // h_sr.data.hop_length,
                                       **h_sr.model)
            utils.load_checkpoint(ckpt_sr, self.speechsr, None)
            self.speechsr.eval()
            self.hps_sr = h_sr
        else:
            self.speechsr = None

        self.denoiser = MPNet(hps_denoiser)
        state_dict = load_checkpoint(ckpt_denoiser, 'cpu')
        self.denoiser.load_state_dict(state_dict['generator'])
        self.denoiser.eval()

        self.hps = hps
        self.hps_denoiser = hps_denoiser
        self.hps_t2w2v = hps_t2w2v

    def vc_to_file(self, audio, save_path, speaker_prompt):

        source_audio, sample_rate = torchaudio.load(audio)
        if sample_rate != 16000:
            source_audio = torchaudio.functional.resample(source_audio, sample_rate, 16000,
                                                          resampling_method="kaiser_window")
        p = (source_audio.shape[-1] // 1280 + 1) * 1280 - source_audio.shape[-1]
        source_audio = torch.nn.functional.pad(source_audio, (0, p), mode='constant').data

        try:
            f0 = get_yaapt_f0(source_audio.numpy())
        except:
            f0 = np.zeros((1, 1, source_audio.shape[-1] // 80))
            f0 = f0.astype(np.float32)
            f0 = f0.squeeze(0)

        ii = f0 != 0
        f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()

        y_pad = F.pad(source_audio, (40, 40), "reflect")
        x_w2v = self.w2v(y_pad.to(self.device))
        x_length = torch.LongTensor([x_w2v.size(2)]).to(self.device)

        # Prompt load
        target_audio, sample_rate = torchaudio.load(speaker_prompt)
        # support only single channel
        target_audio = target_audio[:1, :]
        # Resampling
        if sample_rate != 16000:
            target_audio = torchaudio.functional.resample(target_audio, sample_rate, 16000,
                                                          resampling_method="kaiser_window")
        if self.scale_norm == 'prompt':
            prompt_audio_max = torch.max(target_audio.abs())
        try:
            t_f0 = get_yaapt_f0(target_audio.numpy())
        except:
            t_f0 = np.zeros((1, 1, target_audio.shape[-1] // 80))
            t_f0 = t_f0.astype(np.float32)
            t_f0 = t_f0.squeeze(0)
        j = t_f0 != 0

        f0[ii] = ((f0[ii] * t_f0[j].std()) + t_f0[j].mean()).clip(min=0)
        denorm_f0 = torch.log(torch.FloatTensor(f0 + 1).to(self.device))
        # We utilize a hop size of 320 but denoiser uses a hop size of 400 so we utilize a hop size of 1600
        ori_prompt_len = target_audio.shape[-1]
        p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
        target_audio = torch.nn.functional.pad(target_audio, (0, p), mode='constant').data

        # file_name_t = os.path.splitext(os.path.basename(a.target_speech))[0]

        # If you have a memory issue during denosing the prompt, try to denoise the prompt with cpu before TTS
        # We will have a plan to replace a memory-efficient denoiser
        if self.denoise_ratio == 0:
            target_audio = torch.cat([target_audio.to(self.device), target_audio.to(self.device)], dim=0)
        else:
            with torch.no_grad():
                denoised_audio = denoise(target_audio.squeeze(0).to(self.device), self.denoiser, self.hps_denoiser)
            target_audio = torch.cat([target_audio.to(self.device), denoised_audio[:, :target_audio.shape[-1]]], dim=0)

        target_audio = target_audio[:,
                       :ori_prompt_len]  # 20231108 We found that large size of padding decreases a performance so we remove the paddings after denosing.

        trg_mel = self.mel_fn(target_audio.to(self.device))

        trg_length = torch.LongTensor([trg_mel.size(2)]).to(self.device)
        trg_length2 = torch.cat([trg_length, trg_length], dim=0)

        with torch.no_grad():

            ## Hierarchical Speech Synthesizer (W2V, F0 --> 16k Audio)
            converted_audio = \
                self.net_g.voice_conversion_noise_control(x_w2v, x_length, trg_mel, trg_length2, denorm_f0,
                                                          noise_scale=self.noise_scale_vc,
                                                          denoise_ratio=self.denoise_ratio)

            ## SpeechSR (Optional) (16k Audio --> 24k or 48k Audio)
            if self.output_sr == 48000:
                converted_audio = self.speechsr(converted_audio)
            elif self.output_sr == 24000:
                converted_audio = self.speechsr(converted_audio)
            else:
                converted_audio = converted_audio

        converted_audio = converted_audio.squeeze()

        if self.scale_norm == 'prompt':
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * prompt_audio_max
        else:
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * 0.999

        converted_audio = converted_audio.cpu().numpy().astype('int16')

        # file_name2 = "{}.wav".format(file_name_s + "_to_" + file_name_t)
        # output_file = os.path.join(a.output_dir, file_name2)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.output_sr == 48000:
            write(save_path, 48000, converted_audio)
        elif self.output_sr == 24000:
            write(save_path, 24000, converted_audio)
        else:
            write(save_path, 16000, converted_audio)

    def tts_to_file(self, text, save_path, speaker_prompt, **kwargs):

        text = text_to_sequence(str(text), ["english_cleaners2"])
        token = add_blank_token(text).unsqueeze(0).to(self.device)
        token_length = torch.LongTensor([token.size(-1)]).to(self.device)

        # Prompt load
        audio, sample_rate = torchaudio.load(speaker_prompt)

        # support only single channel
        audio = audio[:1, :]
        # Resampling
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window")
        if self.scale_norm == 'prompt':
            prompt_audio_max = torch.max(audio.abs())

        # We utilize a hop size of 320 but denoiser uses a hop size of 400 so we utilize a hop size of 1600
        ori_prompt_len = audio.shape[-1]
        p = (ori_prompt_len // 1600 + 1) * 1600 - ori_prompt_len
        audio = torch.nn.functional.pad(audio, (0, p), mode='constant').data

        # file_name = os.path.splitext(os.path.basename(a.input_prompt))[0]

        # If you have a memory issue during denosing the prompt, try to denoise the prompt with cpu before TTS
        # We will have a plan to replace a memory-efficient denoiser
        if self.denoise_ratio == 0:
            audio = torch.cat([audio.to(self.device), audio.to(self.device)], dim=0)
        else:
            with torch.no_grad():
                denoised_audio = denoise(audio.squeeze(0).to(self.device), self.denoiser, self.hps_denoiser)
            audio = torch.cat([audio.to(self.device), denoised_audio[:, :audio.shape[-1]]], dim=0)

        audio = audio[:,
                :ori_prompt_len]  # 20231108 We found that large size of padding decreases a performance so we remove the paddings after denosing.

        src_mel = self.mel_fn(audio.to(self.device))

        src_length = torch.LongTensor([src_mel.size(2)]).to(self.device)
        src_length2 = torch.cat([src_length, src_length], dim=0)

        ## TTV (Text --> W2V, F0)
        with torch.no_grad():
            w2v_x, pitch = self.text2w2v.infer_noise_control(token, token_length, src_mel, src_length2,
                                                        noise_scale=self.noise_scale_ttv,
                                                        denoise_ratio=self.denoise_ratio)

            src_length = torch.LongTensor([w2v_x.size(2)]).to(self.device)

            ## Pitch Clipping
            pitch[pitch < torch.log(torch.tensor([55]).to(self.device))] = 0

            ## Hierarchical Speech Synthesizer (W2V, F0 --> 16k Audio)
            converted_audio = \
                self.net_g.voice_conversion_noise_control(w2v_x, src_length, src_mel, src_length2, pitch,
                                                     noise_scale=self.noise_scale_vc, denoise_ratio=self.denoise_ratio)

            ## SpeechSR (Optional) (16k Audio --> 24k or 48k Audio)
            if self.output_sr == 48000:
                converted_audio = self.speechsr(converted_audio)
            elif self.output_sr == 24000:
                converted_audio = self.speechsr(converted_audio)
            else:
                converted_audio = converted_audio

        converted_audio = converted_audio.squeeze()

        if self.scale_norm == 'prompt':
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * prompt_audio_max
        else:
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 32767.0 * 0.999

        converted_audio = converted_audio.cpu().numpy().astype('int16')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.output_sr == 48000:
            write(save_path, 48000, converted_audio)
        elif self.output_sr == 24000:
            write(save_path, 24000, converted_audio)
        else:
            write(save_path, 16000, converted_audio)

    def super_resolution_to_file(self, audio, save_path):
        # Prompt load
        audio, sample_rate = torchaudio.load(audio)

        # support only single channel
        audio = audio[:1, :]
        # Resampling
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000, resampling_method="kaiser_window")

        ## SpeechSR (Optional) (16k Audio --> 24k or 48k Audio)
        with torch.no_grad():
            converted_audio = self.speechsr(audio.unsqueeze(1).to(self.device))
            converted_audio = converted_audio.squeeze()
            converted_audio = converted_audio / (torch.abs(converted_audio).max()) * 0.999 * 32767.0
            converted_audio = converted_audio.cpu().numpy().astype('int16')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if self.output_sr == 48000:
            write(save_path, 48000, converted_audio)
        else:
            write(save_path, 24000, converted_audio)

    def forward(self, *args, **kwargs):
        pass
