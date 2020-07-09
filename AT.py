import numpy as np
import yaml
import torch

class AT:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.nepochs = self.config.num_epoches

        at_config_path = getattr(self.config, "at_config", None)
        if not at_config_path:
            self.train = lambda *args, **kwargs: None
            self.schedule = lambda key, value, at_config: value
            return

        at_config_dict = yaml.safe_load(open(at_config_path, 'r'))
        self.at_config = Config()
        for k, v in at_config_dict.items():
            setattr(self.at_config, k, v)

        self.generate = getattr(ATSampleGenerator, f"{self.at_config.generator}_generate", None)
        assert(self.generate is not None, f"Requested AT generator `{self.at_config.generator}' is not supported")

        self.schedule_ = getattr(ATScheduler, f"{self.at_config.scheduler}_schedule", None)
        assert(self.schedule_ is not None, f"Requested AT scheduler `{self.at_config.scheduler}' is not supported")

    def schedule(self, key, value):
        info = {
                "nepochs": self.nepoch,
                "epoch": self.epoch
        }
        return self.schedule_(key, value, info, self.at_config)


    # FIXME: too many arguments
    def train(self, x, optimizer, loss, data, model, loss_fn, lids, device):
        if self.epoch < self.at_config.start_epoch or self.epoch > self.at_config.end_epoch or np.random.binomial(1, self.at_config.prob) == 0:
            return

        # 1. propagate the normal training gradient for AT
        loss.backward()

        # 2. generate AT sample
        x = self.generate(x, self.at_config)
        if lids is not None:
            xx = lids.repeat(B,T,1)
            x[:,:,-1] = xx

        # 3. retrain model and return loss
        _, input_sizes, targets, target_sizes, utt_list = data

        x = x.to(device)
        input_sizes = input_sizes.to(device)
        targets = targets.to(device)
        target_sizes = target_sizes.to(device)

        out = model(x)
        out_len, batch_size, _ = out.size()
        input_sizes = (input_sizes * out_len).long()

        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss *= self.at_config.alpha
        loss /= batch_size

        return loss

    def step(self):
        self.epoch += 1


class ATScheduler:

    @staticmethod
    def pas_schedule(self, key, value, info, at_config):
        supported_schedules = ["lr", "epsilon", "save"]
        requested_info = ["epoch", "nepochs"]
        assert( key in supported_schedules, f"Schedule `{key}' is not supported in PAS" )
        assert( all([condition in info for condition in requested_info]), f"Not all requested_info are provieded for scheduler PAS" )

        if (key == "lr"):
            return value

        if (key == "epsilon"):
            epsilon = min(nepoch / info["nepochs"] * (at_config.pas_percent), 1) * value
            return epsilon

        if (key == "save"):
            return False


class ATSampleGenerator:

    @staticmethod
    def fgsm_generate(x, at_config):
        x_fgsm = x.cpu() + np.sign(x.grad.cpu()) * at_config.epsilon
        return x_fgsm

    @staticmethod
    def random_generate(x, at_config):
        x = x + torch.FloatTensor(at_config.epsilon * np.sign(np.random.rand(x.shape)))
        return x


class Config:
    pass
