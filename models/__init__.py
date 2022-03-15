from .hit_net_sf import HITNet_SF, HITNetXL_SF
from .hit_net_kitti import HITNet_KITTI
from .stereo_net import StereoNet
from .stereo_net_sep import StereoNetSep
from .stereo_net_new import StereoNetNew


def build_model(args):
    if args.model == "HITNet_SF":
        model = HITNet_SF()
    elif args.model == "HITNetXL_SF":
        model = HITNetXL_SF()
    elif args.model == "HITNet_KITTI":
        model = HITNet_KITTI()
    elif args.model == "StereoNet":
        model = StereoNet()
    elif args.model == "StereoNetSep":
        model = StereoNetSep()
    elif args.model == "StereoNetNew":
        model = StereoNetNew()
    else:
        raise NotImplementedError

    return model
