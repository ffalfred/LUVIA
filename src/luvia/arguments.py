import argparse

class LUVIAargs():

    # Dictionary to track which arguments belong to which group
    ARG_GROUPS = {
                "general": ["input", "output", "verbose", "clean_mode"],
                "clean_simple": ["blur_kernel", "blur_sigma",
                        "block_size", "vthresh_C", "min_area", "max_area",
                        "min_aspect", "max_aspect", "min_vertices"],
                "clean_otsa": ["blur_kernel_size", "canny_thresh1", "canny_thresh2",
                            "cc_min_area", "cc_max_area", "contour_min_area", "contour_max_area",
                            "contour_min_vertices", "contour_max_vertices"],
                "hoofv_threshold": ["kernel_size", "iterations"],
                "hoofv_cca": ["filter_angle", "min_area_segment","filter_boxes",
                                    "dilation_kernel", "angle_tolerance"],
                "hoofh": ["sigma", "separation_char"],
                "straw": ["weights", "infer_mode", "length_norm", "beam_width",
                        "num_groups", "diversity_strength", "top_k", "top_p",
                        "temperature", "k", "notransform_input"],
                "tongue": []
                }


    @staticmethod
    def default_args(parser):

        general = parser.add_argument_group("General Settings")
        general.add_argument("-i", "--input", help="Input file")
        general.add_argument("-o", "--output", help="Output folder")
        general.add_argument("--clean_mode", choices=["OTSA", "simple", False], default="OTSA")
        general.add_argument("--rotate_img", default=-90)
        general.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    @staticmethod
    def clean_args(parser):
        clean = parser.add_argument_group("Clean Image Settings")      
        clean.add_argument("--blur_kernel", default=(5, 5), nargs=2, type=int,
                            metavar=("START", "END"),)
        clean.add_argument("--blur_sigma", default=0)
        clean.add_argument("--block_size", default=15)
        clean.add_argument("--vthresh_C", default=3)
        clean.add_argument("--min_area", default=20)
        clean.add_argument("--max_area", default=2000)
        clean.add_argument("--min_aspect", default=0.1)
        clean.add_argument("--max_aspect", default=10.0)
        clean.add_argument("--min_vertices", default=6)

        clean.add_argument("--blur_kernel_size", default=5)
        clean.add_argument("--canny_thresh1", default=50)
        clean.add_argument("--canny_thresh2", default=150)
        clean.add_argument("--cc_min_area", default=20)
        clean.add_argument("--cc_max_area", default=2000)
        clean.add_argument("--contour_min_area", default=20)
        clean.add_argument("--contour_max_area", default=2000)
        clean.add_argument("--contour_min_vertices", default=5)
        clean.add_argument("--contour_max_vertices", default=0.001)    

    @staticmethod
    def hoofh_args(parser):
        hoofv = parser.add_argument_group("Hoof vertical Settings")
        hoofv.add_argument("--hoofh_mode", default="cca", choices=["cca", "threshold"])
        hoofv.add_argument("--filter_angle", default=False)
        hoofv.add_argument("--min_area_segment", default=100)
        hoofv.add_argument("--dilation_kernel", default=(90,10), nargs=2, type=int,
                                            metavar=("START", "END"))
        hoofv.add_argument("--angle_tolerance", default=15)
        hoofv.add_argument("--filter_boxes", default="inside_box")
        hoofv.add_argument("--kernel_size", default=(150,20), nargs=2, type=int,
                                            metavar=("START", "END"))
        hoofv.add_argument("--iterations", default=1)



    @staticmethod
    def hoofv_args(parser):
        hoofh = parser.add_argument_group("Hoof horizontal Settings")
        hoofh.add_argument("--sigma", default=4)
        hoofh.add_argument("--separation_char", default=5)

    @staticmethod
    def tongue_args(parser):
        pass

    @staticmethod
    def straw_args(parser):
        straw = parser.add_argument_group("Straw Settings")
        straw.add_argument("--weights")
        straw.add_argument("--infer_mode", default="diverse_beam")
        straw.add_argument("--length_norm", default=True)
        straw.add_argument("--beam_width", default=3)
        straw.add_argument("--num_groups", default=3)
        straw.add_argument("--diversity_strength", default=0.5)
        straw.add_argument("--top_k", default=0)
        straw.add_argument("--top_p", default=0.9)
        straw.add_argument("--temperature", default=1.0)
        straw.add_argument("--k", default=1)
        straw.add_argument("--dictionary",
                            choices=[False, "match", "POSmatch", "characters"], default=False)
        straw.add_argument("--notransform_input", action='store_false', default=True)
        straw.add_argument("--character", default=False)


    def extract_group_args(args, group_name):
        """Extract arguments from a specific group name."""
        keys = LUVIAargs.ARG_GROUPS.get(group_name, [])
        return {key: getattr(args, key, None) for key in keys}


    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Luvia animal")

        subparsers = parser.add_subparsers(dest="command", required=True)
        ## Main
        main_parser = subparsers.add_parser("main", help="Run the main function")
        LUVIAargs.default_args(main_parser)
        LUVIAargs.clean_args(main_parser)
        LUVIAargs.hoofv_args(main_parser)
        LUVIAargs.hoofh_args(main_parser)
        LUVIAargs.straw_args(main_parser)
        LUVIAargs.tongue_args(main_parser)
        ## Clean
        clean_parser = subparsers.add_parser("clean", help="Run the clean function")

        LUVIAargs.default_args(clean_parser)
        LUVIAargs.clean_args(clean_parser)
        ## Tongue
        tongue_parser = subparsers.add_parser("tongue", help="Run the tongue function")
        LUVIAargs.default_args(tongue_parser)
        LUVIAargs.tongue_args(tongue_parser)
        ## Hoof
        hoof_parser = subparsers.add_parser("hoof", help="Run the hoof function")
        LUVIAargs.default_args(hoof_parser)
        LUVIAargs.hoofv_args(hoof_parser)
        LUVIAargs.hoofh_args(hoof_parser)
        ## Straw
        straw_parser = subparsers.add_parser("straw", help="Run the straw function")
        LUVIAargs.default_args(straw_parser)
        LUVIAargs.straw_args(straw_parser)
        ## Spiral
        spial_parser = subparsers.add_parser("spiral", help="Run the spiral function")
        LUVIAargs.default_args(spial_parser)


        return parser.parse_args()


