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
                "tongue": ["dictionary", "character", "corrected_k", "sel_sentence",
                           "final_sentences","quantile"]
                }


    @staticmethod
    def default_args(parser):

        general = parser.add_argument_group("General Settings")
        general.add_argument("-i", "--input", help="Input file")
        general.add_argument("-o", "--output", help="Output folder")
        general.add_argument("--clean_mode", choices=["OTSA", "simple", False], default="OTSA")
        general.add_argument("--rotate_img", default=-90, type=float)
        general.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")

    @staticmethod
    def clean_args(parser):
        clean = parser.add_argument_group("Clean Image Settings")      
        clean.add_argument("--blur_kernel", default="5,5")
        clean.add_argument("--blur_sigma", default=0, type=float)
        clean.add_argument("--block_size", default=15, type=float)
        clean.add_argument("--vthresh_C", default=3, type=float)
        clean.add_argument("--min_area", default=20, type=float)
        clean.add_argument("--max_area", default=2000, type=float)
        clean.add_argument("--min_aspect", default=0.1, type=float)
        clean.add_argument("--max_aspect", default=10.0, type=float)
        clean.add_argument("--min_vertices", default=6, type=float)

        clean.add_argument("--blur_kernel_size", default=5, type=float)
        clean.add_argument("--canny_thresh1", default=50, type=float)
        clean.add_argument("--canny_thresh2", default=150, type=float)
        clean.add_argument("--cc_min_area", default=20, type=float)
        clean.add_argument("--cc_max_area", default=2000, type=float)
        clean.add_argument("--contour_min_area", default=20, type=float)
        clean.add_argument("--contour_max_area", default=2000, type=float)
        clean.add_argument("--contour_min_vertices", default=5, type=float)
        clean.add_argument("--contour_max_vertices", default=0.001, type=float)    

    @staticmethod
    def hoofh_args(parser):
        hoofv = parser.add_argument_group("Hoof vertical Settings")
        hoofv.add_argument("--hoofh_mode", default="cca", choices=["cca", "threshold"])
        hoofv.add_argument("--filter_angle", default=False)
        hoofv.add_argument("--min_area_segment", default=100, type=float)
        hoofv.add_argument("--dilation_kernel", default="90,10")
        hoofv.add_argument("--angle_tolerance", default=15, type=float)
        hoofv.add_argument("--filter_boxes", default="inside_box")
        hoofv.add_argument("--kernel_size", default="150,20")
        hoofv.add_argument("--iterations", default=1, type=float)



    @staticmethod
    def hoofv_args(parser):
        hoofh = parser.add_argument_group("Hoof horizontal Settings")
        hoofh.add_argument("--sigma", default=4, type=float)
        hoofh.add_argument("--separation_char", default=5, type=float)

    @staticmethod
    def tongue_args(parser):
        tongue = parser.add_argument_group("Tongue Settings")
        tongue.add_argument("--dictionary",
                            choices=[False, "vanilla", "equal_POS", "character_POS"], default="character_POS")
        tongue.add_argument("--character", default="random")
        tongue.add_argument("--corrected_k", default=5, type=int)
        tongue.add_argument("--sel_sentence", choices=["random", "best", "quantile"], default="quantile")
        tongue.add_argument("--quantile", choices=["5th", "10th", "25th", "50th", "75th",
                                                   "90th", "95th", "100th"], default="5th")
        tongue.add_argument("--final_sentences", default=2, type=int)

    @staticmethod
    def straw_args(parser):
        straw = parser.add_argument_group("Straw Settings")
        straw.add_argument("--weights", default="random")
        straw.add_argument("--infer_mode", default="diverse_beam")
        straw.add_argument("--length_norm", action="store_true")
        straw.add_argument("--beam_width", default=3, type=int)
        straw.add_argument("--num_groups", default=3, type=int)
        straw.add_argument("--diversity_strength", default=0.5, type=float)
        straw.add_argument("--top_k", default=0, type=float)
        straw.add_argument("--top_p", default=0.9, type=float)
        straw.add_argument("--temperature", default=1.0, type=float)
        straw.add_argument("--k", default=1, type=int)
        straw.add_argument("--notransform_input", action='store_false', default=True)

    @staticmethod
    def extract_group_args(args, group_name):
        """Extract arguments from a specific group name."""
        keys = LUVIAargs.ARG_GROUPS.get(group_name, [])
        return {key: getattr(args, key, None) for key in keys}
    
    @staticmethod
    def fix_doublevalue(argument):
        arguments_split = argument.split(",")
        argument_lst = []
        for n in arguments_split:
            argument_lst.append(int(n.strip()))
        return tuple(argument_lst)

    
    @staticmethod
    def fix_args(arguments_parse):
        if hasattr(arguments_parse, "blur_kernel"):
            arguments_parse.blur_kernel = LUVIAargs.fix_doublevalue(arguments_parse.blur_kernel)
        if hasattr(arguments_parse, "dilation_kernel"):
            arguments_parse.dilation_kernel = LUVIAargs.fix_doublevalue(arguments_parse.dilation_kernel)
        if hasattr(arguments_parse, "kernel_size"):
            arguments_parse.kernel_size = LUVIAargs.fix_doublevalue(arguments_parse.kernel_size)
        return arguments_parse

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Luvia animal",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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


        arguments_parse = parser.parse_args()
        arguments_parse = LUVIAargs.fix_args(arguments_parse)
        return arguments_parse


