import argparse

from luvia.config import (
    GROUP_MODELS, add_argparse_from_model,
    GeneralConfig, CleanSimpleConfig, CleanOtsaConfig,
    HoofVCCAConfig, HoofVThresholdConfig, HoofHConfig,
    StrawConfig, TongueConfig, HordeConfig,
)


class LUVIAargs():

    # Argument-group keys consumed by ``extract_group_args``. Derived from the
    # Pydantic models in luvia.config so the group names, field names and
    # field types are a single source of truth -- no more keeping ARG_GROUPS
    # in sync with argparse by hand.
    ARG_GROUPS = {
        group: list(model.model_fields.keys())
        for group, model in GROUP_MODELS.items()
    }

    @staticmethod
    def default_args(parser):
        add_argparse_from_model(
            parser.add_argument_group("General Settings"),
            GeneralConfig,
        )

    @staticmethod
    def clean_args(parser):
        group = parser.add_argument_group("Clean Image Settings")
        add_argparse_from_model(group, CleanSimpleConfig)
        add_argparse_from_model(group, CleanOtsaConfig)

    @staticmethod
    def hoofh_args(parser):
        # hoofh_mode is the selector that picks between cca and threshold;
        # it's not part of either downstream config so it stays hand-written.
        group = parser.add_argument_group("Hoof vertical Settings")
        group.add_argument("--hoofh_mode", default="cca",
                           choices=["cca", "threshold"])
        add_argparse_from_model(group, HoofVCCAConfig)
        add_argparse_from_model(group, HoofVThresholdConfig)

    @staticmethod
    def hoofv_args(parser):
        add_argparse_from_model(
            parser.add_argument_group("Hoof horizontal Settings"),
            HoofHConfig,
        )

    @staticmethod
    def tongue_args(parser):
        add_argparse_from_model(
            parser.add_argument_group("Tongue Settings"),
            TongueConfig,
        )

    @staticmethod
    def straw_args(parser):
        add_argparse_from_model(
            parser.add_argument_group("Straw Settings"),
            StrawConfig,
        )

    @staticmethod
    def horde_args(parser):
        add_argparse_from_model(
            parser.add_argument_group("Horde Settings"),
            HordeConfig,
        )

    @staticmethod
    def extract_group_args(args, group_name):
        """Extract a group's args from a Namespace + validate via Pydantic.

        Pulls the raw values from the argparse Namespace, then validates and
        type-coerces them through the matching Pydantic model in
        ``luvia.config.GROUP_MODELS``. Return type stays ``dict`` so the rest
        of the dispatcher is unchanged.
        """
        keys = LUVIAargs.ARG_GROUPS.get(group_name, [])
        raw = {key: getattr(args, key, None) for key in keys
               if getattr(args, key, None) is not None}
        model_class = GROUP_MODELS.get(group_name)
        if model_class is None:
            return {key: getattr(args, key, None) for key in keys}
        return model_class.model_validate(raw).model_dump()

    @staticmethod
    def fix_doublevalue(argument):
        return tuple(int(n.strip()) for n in argument.split(","))

    @staticmethod
    def fix_args(arguments_parse):
        # The three Tuple[int, int] fields are taken as comma-separated strings
        # at the CLI; convert them to actual tuples before downstream code
        # (and Pydantic validation) sees them.
        for name in ("blur_kernel", "dilation_kernel", "kernel_size"):
            if hasattr(arguments_parse, name):
                setattr(arguments_parse, name,
                        LUVIAargs.fix_doublevalue(getattr(arguments_parse, name)))
        return arguments_parse

    @staticmethod
    def main(argv=None):
        # argv=None preserves CLI behavior (reads sys.argv); passing a list lets
        # the GUI (PipelineWorker) drive the same parser programmatically.
        parser = argparse.ArgumentParser(
            description="Luvia animal",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        main_parser = subparsers.add_parser("main", help="Run the main function")
        LUVIAargs.default_args(main_parser)
        LUVIAargs.clean_args(main_parser)
        LUVIAargs.hoofv_args(main_parser)
        LUVIAargs.hoofh_args(main_parser)
        LUVIAargs.straw_args(main_parser)
        LUVIAargs.tongue_args(main_parser)

        horde_parser = subparsers.add_parser("horde", help="Run the horde function")
        LUVIAargs.default_args(horde_parser)
        LUVIAargs.horde_args(horde_parser)
        LUVIAargs.clean_args(horde_parser)
        LUVIAargs.hoofv_args(horde_parser)
        LUVIAargs.hoofh_args(horde_parser)
        LUVIAargs.straw_args(horde_parser)
        LUVIAargs.tongue_args(horde_parser)

        arguments_parse = parser.parse_args(argv)
        arguments_parse = LUVIAargs.fix_args(arguments_parse)
        return arguments_parse
