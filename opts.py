#参数配置
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
#指定配置文件的地方
    parser.add_argument('--config', is_config_file=True,default='configs/fern.txt',
                        help='config file path')
    # 本次实验的名称,作为log中文件夹的名字
    parser.add_argument("--expname", type=str,default='./blender_paper_fern',
                        help='experiment name')
    # 输出目录
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store ckpts and logs')
    # 指定数据集的目录
    parser.add_argument("--datadir", type=str, default='./data/nerf_llff_data/fern',
                        help='input data directory')

    # training options神经网络相关参数
    # 全连接的层数，训练空间位置点的层数8层
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    # 网络宽度，全连接
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')

    # 精细网络的全连接层数
    # 默认精细网络的深度和宽度与粗糙网络是相同的
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')

    # 这里的batch size，指的是光线的数量,像素点的数量
    # N_rand 配置文件中是1024
    # 32*32*4=4096
    # 800*800/4096=156 400*400/1024=156
    #在一次迭代中使用像素点的数量，默认给出的是4096,config文件中指定1024
    parser.add_argument("--N_rand", type=int, default=32 * 32 ,
                        help='batch size (number of random rays per gradient step)')
    # 学习率
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    #如果像素点的值大于chunk值会在运行的时候分批次处理，定义光线数量
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')

    # 网络中处理的点的数量，如果显存不够可以把这个值调小，调小之后也会对这个进行分批次的处理
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    # 合成的数据集一般都是True, 每次只从一张图片中选取随机光线
    # 真实的数据集一般都是False, 图形先混在一起
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')

    # 不加载权重，不使用之前保存的权重文件
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # 粗网络的权重文件的位置
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    #粗网络一条光线上采样点的数量
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    #精细网络使用采样点的数量
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    #是否在采样点附近有一个扰动变化
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 不适用视角数据
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # 0 使用位置编码，-1 不使用位置编码，下面是位置编码相关代码
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')

    # L=10空间位置参数
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    # L=4视角的参数
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # 仅进行渲染
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 渲染test数据集
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 下采样的倍数
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    # 中心裁剪的训练轮数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=0.5, help='fraction of img taken for central crops')

    # dataset options
    # 标识了使用何种数据格式的数据集
    parser.add_argument("--dataset_type", type=str, default='blender',
                        help='options: llff / blender / deepvoxels')

    # 对于大的数据集，test和val数据集，只使用其中的一部分数据，测试数据集较大时，每隔8张采集一张，并非选取所有的图片
    #如果在某个数据集下，test数据集图片比较多的话，每隔8张采集一张
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    # 白色背景
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')

    # 使用一半分辨率，一般在合成数据集下设置，进行一半的下采样，加快训练速度，在真实数据集下一般不设置
    parser.add_argument("--half_res", action='store_false',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    #下采样的倍数，训练的时候
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')

    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    # log输出的频率
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 保存模型的频率
    # 每隔1w保存一个
    parser.add_argument("--i_weights", type=int, default=1000,
                        help='frequency of weight ckpt saving')
    # 执行测试集渲染的频率
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    # 执行渲染视频的频率
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser
