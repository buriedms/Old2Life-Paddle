import os
import paddle
import sys


class BaseModel(paddle.nn.Layer):
    def name(self):
        return "BaseModel"

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = "%s_net_%s.pdparams" % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        paddle.save(network.state_dict(), save_path)

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = "%s_optimizer_%s.pdparams" % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        paddle.save({'parameters': optimizer.state_dict(), 'lr': optimizer.get_lr()}, save_path)

    def load_optimizer(self, optimizer, optimizer_label, epoch_label, save_dir=""):
        save_filename = "%s_optimizer_%s.pdparams" % (epoch_label, optimizer_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)

        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
        else:
            optimizer.set_state_dict(paddle.load(save_path)['parameters'])
            optimizer.set_lr(paddle.load(save_path)['lr'])
            print('optimizer import path:', save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=""):
        save_filename = "%s_net_%s.pdparams" % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
        else:
            try:
                network.set_state_dict(paddle.load(save_path))
                print('network import path:',save_path)
            except:
                raise NotImplementedError
                pretrained_dict = paddle.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.set_state_dict(pretrained_dict)
                    # if self.opt.verbose:
                    print(
                        "Pretrained network %s has excessive layers; Only loading layers that are used"
                        % network_label
                    )
                except:
                    print(
                        "Pretrained network %s has fewer layers; The following are not initialized:"
                        % network_label
                    )
                    for k, v in pretrained_dict.items():
                        if v.shape == model_dict[k].shape:
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set

                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.shape != pretrained_dict[k].shape:
                            not_initialized.add(k.split("."))

                    print(sorted(not_initialized))
                    network.set_state_dict(model_dict)

    def update_learning_rate():
        pass
