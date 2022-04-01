from hydra import compose, initialize
from sys import argv
initialize(config_path="config")
hprams = compose(config_name="configs", overrides=argv[1:])


def get_audpipe_params():
    return {
        'sampling_rate': hprams.data.sampling_rate,
        'n_mel_channels': hprams.data.n_mels,
        'win_length': hprams.data.win_length,
        'hop_length': hprams.data.hop_length,
        'n_time_masks': hprams.data.time_mask.number_of_masks,
        'ps': hprams.data.time_mask.ps,
        'max_freq_mask': hprams.data.freq_mask.max_freq
    }


def get_model_params(vocab_size: int):
    return {
        'enc_params': hprams.model.enc,
        'dec_params': dict(
            **hprams.model.dec,
            vocab_size=vocab_size
            )
    }


def get_optim_params():
    betas = [
        hprams.training.optim.beta1,
        hprams.training.optim.beta2,
        ]
    return dict(
        **hprams.training.optim,
        betas=betas
        )
