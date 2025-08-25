from tqdm import tqdm

from luvia.arguments import LUVIAargs
from luvia.hoof.hoof import Hoofs, Hoof_HThresh, Hoof_VThresh, ShorthandSegmenter
from luvia.eyes.eyes import Eyes_Contour_Clean, Eyes_OTSU_Clean
from luvia.straw.straw import Straw
from luvia.tongue.tongue import Tongue

from luvia.utils.image_utils import ImageUtils
from luvia.utils.output_utils import OutUtils

class LUVIA:

    def __init__(self, out_folder, mode="main"):
        if mode in ["main", "tongue", "hoof", "straw", "spiral"]:
            self.mode = mode
            if mode != "main":
                ValueError("Not yet implemented!!!")
        else:
            ValueError("Mode {} not available".format(mode))

        self.out_module = OutUtils(base_folder=out_folder)

    def main(self, image_path, rotate_image=0, clean_image="OTSA",
                clean_args=dict(), extract_images="cca", extract_lines_args=dict(),
                extract_character_args=dict(), infer_model_args=dict(),
                sentences_model_args=dict()):

        image = ImageUtils.load_image(image_path=image_path)
        self.out_module.save_image(image, "original")

        if not clean_image:
            cleaned_image = image
        else:
            if clean_image == "simple":
                cleaned_image = Eyes_Contour_Clean.extract_original_strokes(image=image,
                                                                            **clean_args)
            else:
                cleaned_image = Eyes_OTSU_Clean.extract_shorthand_strokes(image=image,
                                                                            **clean_args)
        self.out_module.save_image(cleaned_image, "cleaned")

        image_rotated = ImageUtils.rotate_image(cleaned_image, angle=rotate_image)
        self.out_module.save_image(image_rotated, "rotated")


        if extract_images == "cca":
            angle_filtered = extract_lines_args.pop("filter_angle")
            filter_boxes = extract_lines_args.pop("filter_boxes")

            segmenter = ShorthandSegmenter(**extract_lines_args)
            lines = segmenter.extract_groups(image_rotated, filter_boxes)
            if angle_filtered:
                lines = segmenter.filter_by_angle(angle_filtered)
            image_contours = segmenter.draw_bounding_boxes(image_rotated)

        elif extract_images == "threshold":
            image_contours, lines = Hoof_HThresh.extract_lines(image_rotated, **extract_lines_args)
        else:
            ValueError("That option is not available")

        self.out_module.save_image(image_contours, "contours")

        straw = Straw()
        weights_straw = infer_model_args.pop("weights")
        notransform_input = infer_model_args.pop("notransform_input")
        straw.load_model(weights_straw)
        line_count = 0
        dictionary = sentences_model_args.pop("dictionary")
        character = sentences_model_args.pop("character")
        corrected_k = sentences_model_args.pop("corrected_k")
        sel_sentence = sentences_model_args.pop("sel_sentence")
        quantile = sentences_model_args.pop("quantile")
        final_sentences = sentences_model_args.pop("final_sentences")

        sentences_demo = []
        for line in tqdm(lines):
            self.out_module.save_image(line, folder="line",
                                prefix="image_line-{}".format(line_count))
            image_color, characters, params_vproj = Hoof_VThresh.vertical_projection_segmentation(line, **extract_character_args)
            self.out_module.save_projection_image(image_color, prefix="image_vertical_projection_line_{}".format(line_count),
                            projection=params_vproj["projection"], minima=params_vproj["minima"],
                            maxima=params_vproj["maxima"])
            char_count = 0
            for char in characters:
                self.out_module.save_image(char, folder="character",
                                    prefix="image_line-{}_character-{}".format(line_count, char_count))
                char_count += 1
            dataloader = straw.load_data(characters, notransform_input)
            results = straw.infer_model(dataloader, **infer_model_args)
            outputs = []
            for k, val in results.items():
                self.out_module.plot_feature_maps(activation=val["act1"], prefix="cnn_act1_line-{}-{}".format(line_count, k))
                self.out_module.plot_feature_maps(activation=val["act2"], prefix="cnn_act2_line-{}-{}".format(line_count, k))
                self.out_module.maximally_activated_patches(activation=val["act1"], prefix="cnn_actMAX1_line-{}-{}".format(line_count, k))
                self.out_module.plot_filters(layer_weights=val["conv1"], prefix="cnn_act1_line-{}-{}".format(line_count, k))
                self.out_module.plot_filters(layer_weights=val["conv2"], prefix="cnn_act2_line-{}-{}".format(line_count, k))
                self.out_module.plot_saliency(saliency=val["saliency"], prefix="cnn_saliency_line-{}-{}".format(line_count, k))
                self.out_module.plot_sensitivity(sensitivity=val["sensitivity"], prefix="cnn_sensitivity_line-{}-{}".format(line_count, k))
                self.out_module.plot_guidedbackprop(gb_grad=val["gb_grad"], prefix="cnn_guidedbackprop_line-{}-{}".format(line_count, k))
                outputs.append(val["output"])
            if len(outputs) == 0:
                print("Sentence {} doesnt have any character".format(line_count))
                line_count += 1
                continue
            tongue = Tongue(match_mode=dictionary, character=character)
            refined_word_buckets = tongue.finetune_inference(outputs)
            proposed_sentences = tongue.create_sentences(refined_word_buckets)
            corrected_sentences = tongue.correct(proposed_sentences, correct_k=corrected_k)
            analyzed_sentences = tongue.analyze_sentences(corrected_sentences)
            quantiled_sentences = tongue.get_sentence(analyzed_sentences,mode=sel_sentence, quantile=quantile,
                                                        k=final_sentences)
            for sentence in quantiled_sentences:
                sentences_demo.append(sentence["sentence"])
            line_count += 1
        self.out_module.create_pdfresults(sentences_demo)

def main():
    largs = LUVIAargs.main()
    l = LUVIA(out_folder=largs.output, mode=largs.command)
    if largs.command == "main":
        if largs.clean_mode == "simple":
            clean_args = LUVIAargs.extract_group_args(largs, "clean_simple")
        elif largs.clean_mode == "OTSA":
            clean_args = LUVIAargs.extract_group_args(largs, "clean_OTSA")
        else:
            clean_args = False 
        if largs.hoofh_mode == "cca":
            hoofh_args = LUVIAargs.extract_group_args(largs, "hoofv_cca")
        elif largs.hoofh_mode == "threshold":
            hoofh_args = LUVIAargs.extract_group_args(largs, "hoofv_threshold")
        else:
            hoofh_args = False

        l.main(image_path = largs.input, rotate_image=largs.rotate_img,
                clean_image=largs.clean_mode, clean_args=clean_args,
                extract_images=largs.hoofh_mode, extract_lines_args=hoofh_args,
                extract_character_args=LUVIAargs.extract_group_args(largs, "hoofh"),
                infer_model_args=LUVIAargs.extract_group_args(largs, "straw"),
                sentences_model_args=LUVIAargs.extract_group_args(largs, "tongue"))
            
if __name__== "__main__":
    main()
    

            


