# coding:utf8
import warnings
import torch as t


class DefaultConfig(object):
    mark = 'a-1'

    env = 'default'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'UNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = '/home/lz/sgd/data/'  # 训练集存放路径
    test_data_root = train_data_root   # 测试集存放路径
    load_model_path = '/home/lz/sgd/checkpoints/UNet_red_16/a-1--0.05--4/0080--0.7725--0.7498.pth'

    batch_size = 4    # batch size
    use_gpu = True    # user GPU or not
    num_workers = 8  # how many workers for loading data
    print_freq = 20   # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = None

    max_epoch = 100
    lr = 0.01 * 5  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数

    device_ids = [0]

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
