class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 10575 
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'
    seed = 2023
    display = False
    finetune = False

    
    train_root = "casia_160x160_masked_rand.txt"
    # val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/home/hai/workplace/lfw-align-128'

    checkpoints_path = 'checkpoints/model_50_{}'.format(seed)
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/model_50_2022/resnet50_138.pth' #'resnet18_110.pth' #'backbone.pth' #'checkpoints/resnet18_110.pth'
    save_interval = 1

    train_batch_size = 128  # batch size
    test_batch_size = 77

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 64  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 300
    lr = 1e-2  # initial learning rate
    lr_step = 30
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
