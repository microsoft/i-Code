import torch
from core.models.audioldm.latent_diffusion.ema import *
from core.models.audioldm.variational_autoencoder.modules import Encoder, Decoder
from core.models.audioldm.variational_autoencoder.distributions import DiagonalGaussianDistribution

from core.models.audioldm.hifigan.utilities import get_vocoder, vocoder_infer
        
from core.models.audioldm.audio.tools import wav_to_fbank
from core.models.audioldm.audio.stft import TacotronSTFT
def ddconfig():
    return {
#     "first_stage_config": {
#         "base_learning_rate": 4.5e-05,
#         "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
#         "params": {
#             "monitor": "val/rec_loss",
#             "image_key": "fbank",
#             "subband": 1,
#             "embed_dim": 8,
#             "time_shuffle": 1,
#             "ddconfig": {
                "double_z": True,
                "z_channels": 8,
                "resolution": 256,
                "downsample_time": False,
                "in_channels": 1,
                "out_ch": 1,
                "ch": 128,
                "ch_mult": [1, 2, 4],
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "dropout": 0.0,
#             },
#         },
#     },
}
   
class AudioAutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig=ddconfig(),
        lossconfig=None,
        image_key="fbank",
        embed_dim=8,
        time_shuffle=1,
        subband=1,
        ckpt_path=None,
        reload_from_ckpt=None,
        ignore_keys=[],
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.vocoder = get_vocoder(None, "cpu")
        self.embed_dim = embed_dim

        self.fn_STFT = TacotronSTFT()

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

    def encode(self, x, time=10.0):
#         if next(self.parameters()).dtype = torch.float16:
#             self = self.float()
        temp_dtype = x.dtype
        x = wav_to_fbank(
                x.float(), target_length=int(time * 102.4), fn_STFT=self.fn_STFT.float()
            ).to(x.device).to(temp_dtype)
        x = self.freq_split_subband(x)
        h = self.encoder(x)
        moments = self.quant_conv(h)
#         .to(temp_dtype)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        dec = self.freq_merge_subband(dec)
        return dec

    def decode_to_waveform(self, dec):
        dec = dec.squeeze(1).permute(0, 2, 1)
        wav_reconstruction = vocoder_infer(dec, self.vocoder)
        return wav_reconstruction

    def forward(self, input, sample_posterior=True):
        
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        if self.flag_first_run:
            print("Latent size: ", z.size())
            self.flag_first_run = False

        dec = self.decode(z)

        return dec, posterior

    def freq_split_subband(self, fbank):
        if self.subband == 1 or self.image_key != "stft":
            return fbank

        bs, ch, tstep, fbins = fbank.size()

        assert fbank.size(-1) % self.subband == 0
        assert ch == 1

        return (
            fbank.squeeze(1)
            .reshape(bs, tstep, self.subband, fbins // self.subband)
            .permute(0, 2, 1, 3)
        )

    def freq_merge_subband(self, subband_fbank):
        if self.subband == 1 or self.image_key != "stft":
            return subband_fbank
        assert subband_fbank.size(1) == self.subband  # Channel dimension
        bs, sub_ch, tstep, fbins = subband_fbank.size()
        return subband_fbank.permute(0, 2, 1, 3).reshape(bs, tstep, -1).unsqueeze(1)