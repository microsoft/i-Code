import torch
import torch.nn as nn
from core.models.audioldm.clap.open_clip import create_model
from core.models.audioldm.clap.training.data import get_audio_features
import torchaudio
from transformers import RobertaTokenizer
import torch.nn.functional as F


class CLAPAudioEmbeddingClassifierFreev2(nn.Module):
    def __init__(
        self,
        pretrained_path="",
        key="waveform",
        sampling_rate=16000,
        embed_mode="audio",
        unconditional_prob=0.1,
        random_mute=False,
        max_random_mute_portion=0.5,
        training_mode=True,
        joint_embed_shape=512,
        embed_shape=512,
        num_layers=12,
        depths=[2, 2, 6, 2],
        amodel="HTSAT-tiny",
    ):
        super().__init__()

        self.key = key
        self.device = "cpu"
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.embed_mode = embed_mode
        self.embed_mode_orig = embed_mode
        self.sampling_rate = sampling_rate
        self.unconditional_prob = unconditional_prob
        self.random_mute = random_mute
        self.joint_embed_shape = joint_embed_shape
        self.max_random_mute_portion = max_random_mute_portion
        self.training_mode = training_mode
        self.model, self.model_cfg = create_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
            joint_embed_shape=self.joint_embed_shape,
        )

    def get_unconditional_condition(self, batchsize):
        self.unconditional_token = self.model.get_text_embedding(
            self.tokenizer(["", ""])
        )[0:1]
        return torch.cat([self.unconditional_token.unsqueeze(0)] * batchsize, dim=0)

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def _random_mute(self, waveform):
        # waveform: [bs, t-steps]
        t_steps = waveform.size(-1)
        for i in range(waveform.size(0)):
            mute_size = int(
                self.random_uniform(0, end=int(t_steps * self.max_random_mute_portion))
            )
            mute_start = int(self.random_uniform(0, t_steps - mute_size))
            waveform[i, mute_start : mute_start + mute_size] = 0
        return waveform

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        with torch.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform.cuda())
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, dim=2)
            return similarity.squeeze()

    def forward(self, batch, key=None):

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        if self.embed_mode == "audio":
            audio_dict_list = []
            assert (
                self.sampling_rate == 16000
            ), "We only support 16000 sampling rate"
            # batch: [bs, 1, t-samples]
            batch = torchaudio.functional.resample(
                batch, orig_freq=self.sampling_rate, new_freq=48000
            )
            
            for waveform in self.batch_to_list(batch):
                audio_dict = {}
                audio_dict = get_audio_features(
                    audio_dict,
                    waveform.squeeze(),
                    480000,
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                audio_dict_list.append(audio_dict)
            # [bs, 512]

            embed = self.model.get_audio_embedding(audio_dict_list)

        embed = embed.unsqueeze(1)

        # [bs, 1, 512]
        return embed
