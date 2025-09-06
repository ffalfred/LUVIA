from tqdm import tqdm
import os, shutil
import numpy as np
import random
import json
from datetime import datetime


from luvia.arguments import LUVIAargs
from luvia.hoof.hoof import Hoofs, Hoof_HThresh, Hoof_VThresh, ShorthandSegmenter
from luvia.eyes.eyes import Eyes_Contour_Clean, Eyes_OTSU_Clean
from luvia.straw.straw import Straw
from luvia.tongue.tongue import Tongue

from luvia.utils.image_utils import ImageUtils
from luvia.utils.output_utils import OutUtils

class LUVIA:

    def __init__(self, inverted_img, out_folder, mode="main"):
        if mode in ["main", "tongue", "hoof", "straw", "horde"]:
            self.mode = mode
            if mode != "main" or mode != "horde":
                ValueError("Not yet implemented!!!")
        else:
            ValueError("Mode {} not available".format(mode))

        self.out_module = OutUtils(base_folder=out_folder, mode=mode)
        self.inverted_img = not inverted_img
        self.number_proc = 0

    def first_step(self, image_path, invert=False):
        image = ImageUtils.load_image(image_path=image_path)
        if invert:
            image = ImageUtils.invert_image(image)
        self.out_module.save_image(image, prefix=str(self.number_proc),
                                    suffix="original", inverse=self.inverted_img)

        return image
    
    def _clean_image(self, image, clean_image, clean_args):
        if clean_image == "simple":
            cleaned_image = Eyes_Contour_Clean.extract_original_strokes(image=image,
                                                                        **clean_args)
        else:
            cleaned_image = Eyes_OTSU_Clean.extract_shorthand_strokes(image=image,
                                                                        **clean_args)
        self.number_proc += 1
        self.out_module.save_image(cleaned_image, prefix=str(self.number_proc), 
                                    suffix="cleaned", inverse=self.inverted_img)
        return cleaned_image

    def _rotate_image(self, image, angle):
        self.number_proc += 1
        image_rotated = ImageUtils.rotate_image(image, angle=angle)
        self.out_module.save_image(image_rotated, prefix=str(self.number_proc),
                                        suffix="rotated", inverse=self.inverted_img)
        return image_rotated
        
    def _extract_sentences(self, image_rotated, extract_sentences, extract_lines_args):

        if extract_sentences == "cca":
            angle_filtered = extract_lines_args.pop("filter_angle")
            filter_boxes = extract_lines_args.pop("filter_boxes")

            segmenter = ShorthandSegmenter(**extract_lines_args)
            lines = segmenter.extract_groups(image_rotated, filter_boxes)
            if angle_filtered:
                lines = segmenter.filter_by_angle(angle_filtered)
            image_contours = segmenter.draw_bounding_boxes(image_rotated)

        elif extract_sentences == "threshold":
            image_contours, lines = Hoof_HThresh.extract_lines(image_rotated, **extract_lines_args)
        else:
            ValueError("That option is not available")
        self.number_proc += 1
        self.out_module.save_image(image_contours, prefix=str(self.number_proc),
                                suffix="contours", inverse=self.inverted_img)

        return image_contours, lines
    
    def _extract_characters(self, line, line_count, extract_character_args):
        self.out_module.save_image(line, folder="line", prefix="",
                            suffix="image_line-{}".format(line_count))
        image_color, characters, params_vproj = Hoof_VThresh.vertical_projection_segmentation(line, **extract_character_args)
        self.out_module.save_projection_image(image_color, prefix="image_vertical_projection_line_{}".format(line_count),
                        projection=params_vproj["projection"], minima=params_vproj["minima"],
                        maxima=params_vproj["maxima"])
        char_count = 0
        for char in characters:
            self.out_module.save_image(char, folder="character",prefix="", 
                                suffix="image_line-{}_character-{}".format(line_count, char_count), inverse=(not self.inverted_img))
            char_count += 1
        return characters
    
    def _translate_characters(self, characters, straw, notransform_input, line_count, infer_model_args):

        dataloader = straw.load_data(characters, notransform_input)
        results = straw.infer_model(dataloader, **infer_model_args)
        outputs = []
        for k, val in results.items():
            key = "line-{}_{}".format(line_count, k.lower().replace(" ", "-"))
            self.out_module.image_paths[key+"_dict"] = {}
            self.out_module.plot_feature_maps(activation=val["act1"], prefix="cnn_featmap1", suffix=key)
            self.out_module.plot_feature_maps(activation=val["act2"], prefix="cnn_featmap2", suffix=key)
            self.out_module.maximally_activated_patches(activation=val["act1"], prefix="cnn_actMAX1",suffix=key)
            self.out_module.plot_filters(layer_weights=val["conv1"], prefix="cnn_act1", suffix=key)
            self.out_module.plot_filters(layer_weights=val["conv2"], prefix="cnn_act2", suffix=key)
            self.out_module.plot_saliency(saliency=val["saliency"], prefix="cnn_saliency", suffix=key)
            self.out_module.plot_sensitivity(sensitivity=val["sensitivity"], prefix="cnn_sensitivity", suffix=key)
            self.out_module.plot_guidedbackprop(gb_grad=val["gb_grad"], prefix="cnn_guidedbackprop", suffix=key)
            self.out_module.plot_allchar_images(suffix=key)
            outputs.append(val["output"])
        self.out_module.plot_allsentence_images(line_num=line_count, amount_charact=len(results))
        return outputs
    
    def _morph_sentence(self, outputs, dictionary, character, corrected_k, sel_sentence,
                        quantile, final_sentences):
        sentences_demo = []
        tongue = Tongue(match_mode=dictionary, character=character)
        refined_word_buckets = tongue.finetune_inference(outputs)
        proposed_sentences = tongue.create_sentences(refined_word_buckets)
        corrected_sentences = tongue.correct(proposed_sentences, correct_k=corrected_k)
        analyzed_sentences = tongue.analyze_sentences(corrected_sentences)
        quantiled_sentences = tongue.get_sentence(analyzed_sentences,mode=sel_sentence, quantile=quantile,
                                                    k=final_sentences)
        for sentence in quantiled_sentences:
            sentences_demo.append(sentence["sentence"])
            extra_metadaa = Tongue.analyze_sentence(sentence["sentence"])
            sentence.update(extra_metadaa)
        return sentences_demo, quantiled_sentences
    
    
    def main(self, image_path, rotate_image=0, clean_image_mode="OTSA",
                clean_args=dict(), extract_images="cca", extract_lines_args=dict(),
                extract_character_args=dict(), infer_model_args=dict(),
                sentences_model_args=dict(), random_pick=False):

        self.image = self.first_step(image_path=image_path, invert=self.inverted_img)

        if not clean_image_mode:
            cleaned_image = self.image
        else:
            cleaned_image = self._clean_image(image=self.image, clean_image=clean_image_mode,
                                                clean_args=clean_args)

        image_rotated = self._rotate_image(image=cleaned_image, angle=rotate_image)

        image_contours, lines = self._extract_sentences(image_rotated=image_rotated,
                                                        extract_sentences=extract_images,
                                                        extract_lines_args=extract_lines_args)

        self.out_module.plot_alltransformations()
        straw = Straw()
        weights_straw = infer_model_args.pop("weights")
        notransform_input = infer_model_args.pop("notransform_input")
        straw.load_model(weights_straw)
        dictionary = sentences_model_args.pop("dictionary")
        character = sentences_model_args.pop("character")
        corrected_k = sentences_model_args.pop("corrected_k")
        sel_sentence = sentences_model_args.pop("sel_sentence")
        quantile = sentences_model_args.pop("quantile")
        final_sentences = sentences_model_args.pop("final_sentences")
        sentences_demo = []
        if random_pick:
            random.shuffle(lines)
        for line_count, line in tqdm(enumerate(lines)):
            characters = self._extract_characters(line, line_count, extract_character_args)
            outputs = self._translate_characters(characters, straw=straw, notransform_input=notransform_input,
                                                line_count=line_count, infer_model_args=infer_model_args)
            if len(outputs) == 0:
                print("Sentence {} doesnt have any character".format(line_count))
                continue
            candidate_sentences, sentences_info = self._morph_sentence(outputs=outputs,
                                                    dictionary=dictionary, character=character,
                                                      corrected_k=corrected_k, sel_sentence=sel_sentence,
                                                      quantile=quantile, final_sentences=final_sentences)
            sentences_demo.append(sentences_info)
            if random_pick:
                break
        print(self.out_module.image_paths)
        self.out_module.create_pdfimage()
        self.out_module.create_pdftranslation(sentences_demo)
        return sentences_demo, self.out_module.output_folder

    def _getstreets(self, folder_streets):
        dict_files = {}
        for files1 in os.listdir(folder_streets):
            pathfile1 = "{}/{}".format(folder_streets, files1)
            if os.path.isdir(pathfile1):
                for files2 in os.listdir(pathfile1):
                    pathfile2 = "{}/{}".format(folder_streets, files2)
                    if os.path.isfile(pathfile2):
                        dict_files[pathfile2.replace("/", ".")] = pathfile2
            elif os.path.isfile(pathfile1):
                dict_files[pathfile1.replace("/", ".")] = pathfile1
        return dict_files
    
    def _write_jsonfile(self, json_path, new_entry):

        with open(json_path, "a") as f:
            f.write(json.dumps(new_entry) + "\n")

    
    def horde(self, folder_streets, clean_args=dict(), extract_lines_args=dict(),
                extract_character_args=dict(), infer_model_args=dict(), sentences_model_args=dict(), 
                limit_loops=False, max_runs=10):
        dict_files = self._getstreets(folder_streets=folder_streets)
        loop_active = True
        count_runs = 0
        rotate_angles = np.arange(-180, 190, 10)
        json_path = "{}/LUVIA_history.jsonl".format(self.out_module.output_folder)
        runs_folder = []
        while True:
            file_key = random.choice(list(dict_files.keys()))
            file_path = dict_files[file_key]
            angle = int(random.choice(rotate_angles))
            
            main_instance = self.__class__(inverted_img=self.inverted_img,
                                            out_folder=self.out_module.output_folder,
                                            mode="main")
            
            sentences, out_folder= main_instance.main(image_path=file_path, rotate_image=angle,
                                        clean_image_mode="OTSA", clean_args=clean_args.copy(),
                                        extract_images="cca", extract_lines_args=extract_lines_args.copy(),
                                        extract_character_args=extract_character_args.copy(),
                                        infer_model_args=infer_model_args.copy(),
                                        sentences_model_args=sentences_model_args.copy(),
                                        random_pick=True)
            runs_folder.append(out_folder)
            entry ={"sentence": sentences[0][0]["sentence"],
                    "location": "{}--56,24".format(file_key),
                    "image": "{}/images/cnn_images/sentence-spectrum.jpg".format(out_folder),
                    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "id": os.path.basename(out_folder)
                    }
            self._write_jsonfile(json_path=json_path, new_entry=entry)
            if len(runs_folder) >=max_runs:
                fold_del = runs_folder.pop(0)

                if os.path.exists(fold_del):
                    try:
                        shutil.rmtree(fold_del)
                        print("Folder deleted successfully.")
                    except Exception as e:
                        print("Error deleting folder:", e)

            ## Clean/delete
            if limit_loops:
                count_runs += 1
                if limit_loops <= count_runs:
                    break
            



def main():
    largs = LUVIAargs.main()
    ## Settings ##
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

    ## Commands ##
    l = LUVIA(inverted_img=largs.invert_image, out_folder=largs.output, mode=largs.command)
    if largs.command == "clean":
        l.clean(clean_image_mode=largs.clean_mode, clean_args=clean_args)
    elif largs.command == "hoof":
        l.hoof(rotate_image=largs.rotate_img, extract_images=largs.hoofh_mode, extract_lines_args=largs.hoofh_args,
               extract_character_args=LUVIAargs.extract_group_args(largs, "hoofh"))
    elif largs.command == "main":
        l.main(image_path=largs.input, rotate_image=largs.rotate_img, clean_image_mode=largs.clean_mode,
                clean_args=clean_args, extract_images=largs.hoofh_mode, extract_lines_args=hoofh_args,
                extract_character_args=LUVIAargs.extract_group_args(largs, "hoofh"),
                infer_model_args=LUVIAargs.extract_group_args(largs, "straw"),
                sentences_model_args=LUVIAargs.extract_group_args(largs, "tongue"))
    elif largs.command == "horde":
        l.horde(folder_streets=largs.folder_streets, clean_args=clean_args, extract_lines_args=hoofh_args,
                extract_character_args=LUVIAargs.extract_group_args(largs, "hoofh"),
                infer_model_args=LUVIAargs.extract_group_args(largs, "straw"),
                sentences_model_args=LUVIAargs.extract_group_args(largs, "tongue"),
                limit_loops=False)
            
if __name__== "__main__":
    main()
    

            


