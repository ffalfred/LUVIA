# PyInstaller spec for LUVIA.
#
# Produces a directory bundle under dist/luvia/ suitable for wrapping into
# an AppImage (Linux), .app (Mac), or .exe installer (Windows). The Linux
# AppImage step lives in packaging/build_appimage.sh.
#
# Run from repo root:
#     pyinstaller --clean packaging/LUVIA.spec
#
# Note: this is Step 1 of executable packaging. ML model weights (GPT-2,
# grammar corrector) are NOT bundled here -- they download on first run.
# True offline operation requires pre-caching the HuggingFace models and
# adding the cache dir to `datas`; deferred to Step 2.

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.abspath(SPEC)) + "/..")
SRC = os.path.join(REPO_ROOT, "src")

# Data files: project assets + library data the runtime expects on disk.
datas = [
    # LUVIA's bundled data — character JSONs, CNN weights, dictionary metadata.
    (os.path.join(SRC, "luvia", "data"), "luvia/data"),
    # GUI assets — logos, spinner frames, reference image.
    (os.path.join(SRC, "luvia_gui", "gifs"), "luvia_gui/gifs"),
    (os.path.join(SRC, "luvia_gui", "data"), "luvia_gui/data"),
]
datas += collect_data_files("en_core_web_sm")
datas += collect_data_files("spacy")
datas += collect_data_files("transformers")
datas += collect_data_files("albumentations")
# ety bundles a SQLite etymology database in its package data; it imports
# via pkg_resources which PyInstaller doesn't pick up on its own.
datas += collect_data_files("ety")

# Pre-downloaded Hugging Face + NLTK models, populated by
# packaging/download_models.py. The build assumes the cache exists; if it
# doesn't, PyInstaller fails loudly here instead of producing a half-working
# bundle that secretly tries to download at first run.
datas.append((os.path.join(REPO_ROOT, "build", "models_cache"), "models_cache"))

# Hidden imports: things PyInstaller's static analysis won't find.
hiddenimports = []
hiddenimports += collect_submodules("luvia")
hiddenimports += collect_submodules("luvia_gui")
hiddenimports += collect_submodules("en_core_web_sm")
hiddenimports += [
    "deepmultilingualpunctuation",
    "deep_translator",
    "ety",
    "wiktionaryparser",
]

block_cipher = None

a = Analysis(
    [os.path.join(SRC, "luvia_gui", "main.py")],
    pathex=[SRC],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        # Test data bundled by these packages but never used at runtime.
        "matplotlib.tests",
        "numpy.tests",
        "scipy.tests",
        "pandas.tests",
        # Transformers vision architectures (LUVIA is text-only on the LM side).
        "transformers.models.beit",
        "transformers.models.blip",
        "transformers.models.blip_2",
        "transformers.models.clip",
        "transformers.models.deit",
        "transformers.models.detr",
        "transformers.models.deformable_detr",
        "transformers.models.donut",
        "transformers.models.dpt",
        "transformers.models.swin",
        "transformers.models.swin2sr",
        "transformers.models.swinv2",
        "transformers.models.vit",
        "transformers.models.vivit",
        "transformers.models.yolos",
        # Transformers audio architectures (LUVIA has no audio path).
        "transformers.models.audio_spectrogram_transformer",
        "transformers.models.encodec",
        "transformers.models.hubert",
        "transformers.models.musicgen",
        "transformers.models.sew",
        "transformers.models.sew_d",
        "transformers.models.speech_to_text",
        "transformers.models.speech_to_text_2",
        "transformers.models.unispeech",
        "transformers.models.unispeech_sat",
        "transformers.models.wav2vec2",
        "transformers.models.wav2vec2_conformer",
        "transformers.models.whisper",
        # scikit-image — replaced by cv2 across the active code paths.
        "skimage",
        # pandas — only used by training / data-prep code that lazy-imports it.
        "pandas",
        # The data-prep module (PolishDB / MakeFrequencies) is offline tooling,
        # not part of the inference pipeline. Excluding it keeps pandas, nltk
        # corpora it pulls, and wiktionaryparser out of the bundle too.
        "luvia.utils_dataset",
        # Punctuation model -- only Tongue.punctuate() uses it and that method
        # is never called from the active pipeline. The import was hoisted
        # inside the method so excluding the package here is safe.
        "deepmultilingualpunctuation",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="luvia",
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SRC, "luvia_gui", "gifs", "signal-2025-08-25-003555_003.png"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=False,
    upx_exclude=[],
    name="luvia",
)
