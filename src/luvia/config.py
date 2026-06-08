"""Typed configuration models for LUVIA's argument groups.

These are the single source of truth for the CLI surface: the argparse parser
in :mod:`luvia.arguments` is *generated* from these models via
:func:`add_argparse_from_model`, and the dispatcher's
``LUVIAargs.extract_group_args`` validates argparse Namespace values through
the matching model before handing them to the pipeline.

For fields whose CLI shape can't be inferred from the Python type alone
(e.g. ``Tuple[int, int]`` exposed as a "5,5" string), the field carries a
``json_schema_extra={"argparse": {...}}`` payload understood by the
converter below.

Future steps may have ``LUVIA.main`` / ``LUVIA.horde`` accept the typed
configs directly instead of dict kwargs, and have the GUI build configs
instead of CLI strings.
"""

import typing
from typing import Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field


class _GroupConfig(BaseModel):
    """Base for all argument-group configs. ``extra='allow'`` so a single
    argparse Namespace can be unpacked per-group without rejecting fields
    that belong to other groups in the same subparser."""

    model_config = ConfigDict(extra="allow")


def _tuple_default_str(t):
    return ",".join(str(x) for x in t)


# --- General (top-level) ----------------------------------------------------

class GeneralConfig(_GroupConfig):
    input: Optional[str] = Field(default=None, json_schema_extra={"argparse": {"short": "-i", "help": "Input file"}})
    output: Optional[str] = Field(default=None, json_schema_extra={"argparse": {"short": "-o", "help": "Output folder"}})
    user: str = Field(default="Anonymous", json_schema_extra={"argparse": {"short": "-u"}})
    inverted_image: bool = Field(default=False, json_schema_extra={"argparse": {"help": "Source image has inverted polarity (light strokes on dark)"}})
    # clean_mode used to also accept the literal False; that path was never
    # reachable from the CLI because argparse delivers strings. Dropping it
    # here matches actual behaviour.
    clean_mode: typing.Literal["OTSA", "simple"] = "OTSA"
    rotate_img: float = -90.0
    verbose: bool = Field(default=False, json_schema_extra={"argparse": {"short": "-v", "help": "Enable verbose mode"}})


# --- Clean ------------------------------------------------------------------

class CleanSimpleConfig(_GroupConfig):
    """Args for :meth:`Eyes_Contour_Clean.extract_original_strokes`."""

    blur_kernel: Tuple[int, int] = Field(default=(5, 5), json_schema_extra={"argparse": {"as_string": True}})
    blur_sigma: float = 0.0
    block_size: int = 15
    vthresh_C: float = 3.0
    min_area: float = 20.0
    max_area: float = 2000.0
    min_aspect: float = 0.1
    max_aspect: float = 10.0
    min_vertices: int = 6


class CleanOtsaConfig(_GroupConfig):
    """Args for :meth:`Eyes_OTSU_Clean.extract_shorthand_strokes`."""

    blur_kernel_size: int = 5
    canny_thresh1: float = 50.0
    canny_thresh2: float = 150.0
    cc_min_area: float = 20.0
    cc_max_area: float = 2000.0
    contour_min_area: float = 20.0
    contour_max_area: float = 2000.0
    contour_min_vertices: int = 5
    # Semantically a Hu-moment shape threshold; CLI name kept for stability.
    contour_max_vertices: float = 0.001


# --- Hoof (line + character segmentation) -----------------------------------

class HoofVThresholdConfig(_GroupConfig):
    """Args for :meth:`Hoof_HThresh.extract_lines` (named hoofv_threshold
    in the dispatcher for historical reasons)."""

    kernel_size: Tuple[int, int] = Field(default=(150, 20), json_schema_extra={"argparse": {"as_string": True}})
    iterations: int = 1


class HoofVCCAConfig(_GroupConfig):
    """Args for :meth:`ShorthandSegmenter.extract_groups` (cca mode)."""

    # Original CLI was bool-ish via untyped argparse default=False. Kept as
    # Optional[float]: None = no filtering, float = filter at that angle.
    # Pre-existing semantics were unreliable here; this is the honest version.
    filter_angle: Optional[float] = Field(default=None, json_schema_extra={"argparse": {"type": "float_or_none"}})
    min_area_segment: float = 100.0
    filter_boxes: typing.Literal["inside_box", "whole_img"] = "inside_box"
    dilation_kernel: Tuple[int, int] = Field(default=(90, 10), json_schema_extra={"argparse": {"as_string": True}})
    angle_tolerance: float = 15.0


class HoofHConfig(_GroupConfig):
    """Args for :meth:`Hoof_VThresh.vertical_projection_segmentation`."""

    sigma: float = 4.0
    separation_char: float = 5.0


# --- Straw (CNN inference) --------------------------------------------------

class StrawConfig(_GroupConfig):
    """Args for :meth:`Straw.infer_model` / :meth:`Straw.load_model`."""

    weights: str = "random"
    infer_mode: str = "diverse_beam"
    length_norm: bool = False
    beam_width: int = 3
    num_groups: int = 3
    diversity_strength: float = 0.5
    top_k: float = 0.0
    top_p: float = 0.9
    temperature: float = 1.0
    k: int = 1
    # store_false on the CLI: default True; passing the flag disables transforms.
    notransform_input: bool = Field(default=True, json_schema_extra={"argparse": {"store_false": True}})


# --- Tongue (language model) ------------------------------------------------

class TongueConfig(_GroupConfig):
    """Args for :meth:`Tongue.finetune_inference` / :meth:`Tongue.get_sentence`."""

    dictionary: typing.Literal["vanilla", "equal_POS", "character_POS"] = "character_POS"
    character: str = "random"
    corrected_k: int = 5
    sel_sentence: typing.Literal["random", "best", "quantile"] = "quantile"
    quantile: typing.Literal["5th", "10th", "25th", "50th", "75th", "90th", "95th", "100th"] = "5th"
    final_sentences: int = 3


# --- Horde (multi-image loop) -----------------------------------------------

class HordeConfig(_GroupConfig):
    folder_streets: Optional[str] = None
    num_workers: int = Field(
        default=1,
        json_schema_extra={"argparse": {
            "help": "Parallel worker processes for the horde loop. 1 = original "
                     "sequential. Each worker holds its own model copy (~1-2 GB "
                     "RAM), so scale based on available memory.",
        }},
    )


# Registry consumed by both the dispatcher (extract_group_args) and the
# argparse generator (in luvia.arguments).
GROUP_MODELS = {
    "general": GeneralConfig,
    "clean_simple": CleanSimpleConfig,
    "clean_otsa": CleanOtsaConfig,
    "hoofv_threshold": HoofVThresholdConfig,
    "hoofv_cca": HoofVCCAConfig,
    "hoofh": HoofHConfig,
    "straw": StrawConfig,
    "tongue": TongueConfig,
    "horde": HordeConfig,
}


# ---------------------------------------------------------------------------
# Composed pipeline config (Phase 5 step 3)
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Combined typed configuration for one LUVIA.main() invocation.

    Replaces the previous dict-of-dicts API (clean_args, extract_lines_args,
    extract_character_args, infer_model_args, sentences_model_args). Built by
    :func:`from_namespace` from a parsed argparse Namespace, or constructed
    programmatically (GUI / tests / scripts).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rotate_img: float = -90.0
    clean_mode: Optional[typing.Literal["OTSA", "simple"]] = "OTSA"
    hoofh_mode: typing.Literal["cca", "threshold"] = "cca"

    clean: Union[CleanSimpleConfig, CleanOtsaConfig, None] = None
    hoof_v: Union[HoofVCCAConfig, HoofVThresholdConfig] = Field(default_factory=HoofVCCAConfig)
    hoof_h: HoofHConfig = Field(default_factory=HoofHConfig)
    straw: StrawConfig = Field(default_factory=StrawConfig)
    tongue: TongueConfig = Field(default_factory=TongueConfig)

    @classmethod
    def from_namespace(cls, ns):
        """Build a PipelineConfig from a parsed argparse Namespace."""

        def _from_ns(model_cls):
            return model_cls.model_validate({
                k: getattr(ns, k)
                for k in model_cls.model_fields.keys()
                if getattr(ns, k, None) is not None
            })

        clean_mode = getattr(ns, "clean_mode", None)
        if clean_mode == "OTSA":
            clean = _from_ns(CleanOtsaConfig)
        elif clean_mode == "simple":
            clean = _from_ns(CleanSimpleConfig)
        else:
            clean = None

        hoofh_mode = getattr(ns, "hoofh_mode", "cca")
        if hoofh_mode == "threshold":
            hoof_v = _from_ns(HoofVThresholdConfig)
        else:
            hoof_v = _from_ns(HoofVCCAConfig)

        return cls(
            rotate_img=getattr(ns, "rotate_img", -90.0),
            clean_mode=clean_mode,
            hoofh_mode=hoofh_mode,
            clean=clean,
            hoof_v=hoof_v,
            hoof_h=_from_ns(HoofHConfig),
            straw=_from_ns(StrawConfig),
            tongue=_from_ns(TongueConfig),
        )


# ---------------------------------------------------------------------------
# argparse generator
# ---------------------------------------------------------------------------

def _argparse_kwargs(name, field_info):
    """Return the kwargs dict to pass to ``parser.add_argument`` for a field."""
    annotation = field_info.annotation
    default = field_info.default
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    extra = (field_info.json_schema_extra or {}).get("argparse", {}) if field_info.json_schema_extra else {}

    if "help" in extra:
        help_text = extra["help"]
    else:
        help_text = None

    # Tuple[int, int] -> CLI takes "5,5" string, fix_doublevalue parses post-hoc.
    if extra.get("as_string") or origin is tuple:
        return {"type": str, "default": _tuple_default_str(default), "help": help_text}

    # Optional[float] with a sentinel "no filter" meaning.
    if extra.get("type") == "float_or_none":
        return {"type": float, "default": default, "help": help_text}

    # bool with explicit store_false override (default True; flag disables).
    if extra.get("store_false"):
        return {"action": "store_false", "default": default, "help": help_text}

    if annotation is bool:
        return {"action": "store_true", "default": default, "help": help_text}

    if annotation in (int, float, str):
        return {"type": annotation, "default": default, "help": help_text}

    # Optional[T] -> T's type, default keeps None.
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and non_none[0] in (int, float, str):
            return {"type": non_none[0], "default": default, "help": help_text}
        if typing.get_origin(non_none[0]) is typing.Literal:
            return {"choices": list(typing.get_args(non_none[0])), "default": default, "help": help_text}
        return {"type": str, "default": default, "help": help_text}

    if origin is typing.Literal:
        return {"choices": list(args), "default": default, "help": help_text}

    return {"default": default, "help": help_text}


def add_argparse_from_model(parser, model_cls, group_title=None):
    """Add argparse arguments to ``parser`` for every field of ``model_cls``.

    ``group_title`` puts the args under an :class:`argparse._ArgumentGroup`
    so ``--help`` keeps its existing visual grouping.
    """
    target = parser.add_argument_group(group_title) if group_title else parser

    for name, field_info in model_cls.model_fields.items():
        kwargs = _argparse_kwargs(name, field_info)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        extra = (field_info.json_schema_extra or {}).get("argparse", {}) if field_info.json_schema_extra else {}
        flags = [extra["short"], "--" + name] if "short" in extra else ["--" + name]
        target.add_argument(*flags, **kwargs)
