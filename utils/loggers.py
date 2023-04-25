import os
import json
import torchvision

class TBlogger():
    def __init__(self, cfg):
        from torch.utils.tensorboard import SummaryWriter
        self.dir = cfg.logdir
        self.path = cfg.logdir # TODO: automatic logdir, eg: sf/sr/sf_4/sigma_3 ...
        self.writer = SummaryWriter(log_dir=self.path)

    def log_metrics(self, metrics, i, prefix=""):
        for k, v in metrics.items():
            self.writer.add_scalar(prefix+k, v, global_step=i)

    def log_metrics_on_same_curve(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def log_images(self, images, i):
        for k, v in images.items():
            self.writer.add_image(k, v.squeeze().clamp(0, 1), global_step=i)

    def log_eval_hparams(self, hparam_dict, metric_dict):
        self.writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
        # save to jsonl file
        hparam_dict.update(metric_dict)
        logpath = os.path.join(self.dir, 'logs.jsonl')
        with open(logpath, 'a+') as f:
            f.write(json.dumps(hparam_dict) + "\n")

class JSONlogger:
    def __init__(self, cfg, logdir, xpname):
        self.dir = logdir
        self.path = os.path.join(logdir, xpname) # TODO: automatic logdir, eg: sf/sr/sf_4/sigma_3 ...
        os.makedirs(self.path, exist_ok=True)

    def log_metrics(self, metrics, i, prefix=""):
        metrics = {prefix+str(k) : v for k, v in metrics.items()}
        metrics.update({'it': i})
        logpath = os.path.join(self.path, 'log_fn.jsonl')
        with open(logpath, 'a+') as f:
            f.write(json.dumps(metrics)+ "\n")

    def log_metrics_on_same_curve(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        metrics = {main_tag+"/"+str(k) : v for k, v in tag_scalar_dict.items()}
        metrics.update({'it': global_step})
        logpath = os.path.join(self.path, 'log_fn.jsonl')
        with open(logpath, 'a+') as f:
            f.write(json.dumps(metrics)+ "\n")

    def log_images(self, images, i):
        for k, im in images.items():
            impath = os.path.join(self.path, f'it_{i:04d}_{k}.png')
            torchvision.utils.save_image(im.detach().cpu(), impath)

    def log_eval_hparams(self, hparam_dict, metric_dict):
        hparam_dict.update(metric_dict)
        logpath = os.path.join(self.dir, 'logs.jsonl')
        with open(logpath, 'a+') as f:
            f.write(json.dumps(hparam_dict) + "\n")