import os
from PIL import Image


def resize_reference_to_match_width(reference_path, dynamic_path, output_path):
    """
    Resize the reference image to match the width of the dynamic image,
    preserving aspect ratio to avoid distortion.
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    if not os.path.exists(dynamic_path):
        raise FileNotFoundError(f"Dynamic image not found: {dynamic_path}")

    with Image.open(dynamic_path) as dyn_img:
        target_width = dyn_img.width

    with Image.open(reference_path) as ref_img:
        original_width, original_height = ref_img.size
        aspect_ratio = original_height / original_width
        new_height = int(target_width * aspect_ratio)
        resized_ref = ref_img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        resized_ref.save(output_path)
        print(f"Resized reference image saved to: {output_path}")


# Example usage
resize_reference_to_match_width(
    reference_path="/home/alfredff/Toke/LUVIA/src/luvia_gui/data/2023_84_40_2_0143_00113914.jpeg",
    dynamic_path="/home/alfredff/Toke/LUVIA/test/luviahorde2/images/image-transformation.jpg",
    output_path="/home/alfredff/Toke/LUVIA/src/luvia_gui/data/2023_84_40_2_0143_00113914_small.jpeg"
)


