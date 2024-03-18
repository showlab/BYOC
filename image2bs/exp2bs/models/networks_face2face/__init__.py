# from models.networks_face2face.base_network import BaseNetwork
# from models.networks_face2face.loss import *
# from models.networks_face2face.discriminator import MultiscaleDiscriminator, ImageDiscriminator
# from models.networks_face2face.generator import ModulateGenerator
# import util.util as util
import torch

from exp2bs.models.networks_face2face.base_network import BaseNetwork
from exp2bs.util import util


def find_network_using_name(target_network_name, filename):
    # CP:target_class_name == "exp2bsleakyclampencoder"
    target_class_name = target_network_name + filename
    # CP:module_name == "models.networks_face2face.encoder"
    module_name = 'exp2bs.models.networks_face2face.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
        "Class %s should be a subclass of BaseNetwork" % network

    return network


def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)
    if is_train:
        netD_cls = find_network_using_name(opt.netD, 'discriminator')
        parser = netD_cls.modify_commandline_options(parser, is_train)

    # remove the audio part
    # netA_cls = find_network_using_name(opt.netA, 'encoder')
    # parser = netA_cls.modify_commandline_options(parser, is_train)

    return parser


def create_network(cls, opt):
    net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net


def define_networks(opt, name, type):
    netG_cls = find_network_using_name(name, type)
    return create_network(netG_cls, opt)


def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator')
    return create_network(netG_cls, opt)


def define_D(opt):
    netD_cls = find_network_using_name(opt.netD, 'discriminator')
    return create_network(netD_cls, opt)


def define_E(opt):
    # there exists only one encoder type
    netE_cls = find_network_using_name(opt.netE, 'encoder')
    return create_network(netE_cls, opt)


def define_V(opt):
    # there exists only one encoder type
    netV_cls = find_network_using_name(opt.netV, 'encoder')
    return create_network(netV_cls, opt)

