from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--input_path', type=str, default='./checkpoints/results/input_path', help='defined input path.')
        parser.add_argument('--meta_path_vox', type=str, default='./misc/demo.csv', help='the meta data path')
        parser.add_argument('--driving_pose', action='store_true', help='driven pose to generate a talking face')
        parser.add_argument('--list_num', type=int, default=0, help='list num')
        parser.add_argument('--fitting_iterations', type=int, default=10, help='The iterarions for fit testing')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--start_ind', type=int, default=0, help='the start id for defined driven')
        parser.add_argument('--list_start', type=int, default=0, help='which num in the list to start')
        parser.add_argument('--list_end', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--save_path', type=str, default='./results/', help='where to save data')
        parser.add_argument('--multi_gpu', action='store_true', help='whether to use multi gpus')
        parser.add_argument('--defined_driven', action='store_true', help='whether to use defined driven')
        parser.add_argument('--gen_video', action='store_true', help='whether to generate videos')
        parser.add_argument('--onnx', action='store_true', help='for tddfa')
        parser.add_argument('--mode', type=str, default='cpu', help='gpu or cpu mode')
        parser.add_argument('--source_id_idx', type=int, default=0, help='the index of source identity')
        parser.add_argument('--target_id_idx', type=int, default=1, help='the index of index identity')
        parser.add_argument('--max_frame_num', type=int, default=100, help='the num of max frame in generation')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

        # parser.add_argument('--exp_basis_folder_real', type=str,
        #                     default="landmark2exp/checkpoints/facerecon/results/test_images/epoch_20_000000",
        #                     help='exp_basis_folder_real')

        # parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        # parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
