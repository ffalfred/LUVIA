from tqdm import tqdm
import os, shutil
import numpy as np
import random
import json
from datetime import datetime


from luvia.arguments import LUVIAargs
from luvia.config import PipelineConfig
from luvia.hoof.hoof import Hoofs, Hoof_HThresh, Hoof_VThresh, ShorthandSegmenter
from luvia.eyes.eyes import Eyes_Contour_Clean, Eyes_OTSU_Clean
from luvia.straw.straw import Straw
from luvia.tongue.tongue import Tongue

from luvia.utils.image_utils import ImageUtils
from luvia.utils.output_utils import OutUtils


class CancelledError(Exception):
    """Raised when the pipeline detects a user-requested cancellation."""


class LUVIA:

    def __init__(self, inverted_img, out_folder, user, mode="main"):
        if mode not in ("main", "horde"):
            raise ValueError("Mode {} not available".format(mode))
        self.mode = mode

        self.inverted_img = inverted_img
        self.number_proc = 0
        self.username = user
        self.out_folder = out_folder

    def first_step(self, image_path, invert=False):
        image = ImageUtils.load_image(image_path=image_path)
        if invert:
            image = ImageUtils.invert_image(image)
        self.out_module.save_image(image, prefix=str(self.number_proc),
                                    suffix="original", inverse=True, angle=-90, general=True)
        return image
    
    def _clean_image(self, image, clean_image, clean_args):
        if clean_image == "simple":
            cleaned_image = Eyes_Contour_Clean.extract_original_strokes(image=image,
                                                                        **clean_args)
        else:
            cleaned_image = Eyes_OTSU_Clean.extract_shorthand_strokes(image=image,
                                                                        **clean_args)
        self.number_proc += 1
        cleaned_image = ImageUtils.add_canvas(image=cleaned_image)
        self.out_module.save_image(cleaned_image, prefix=str(self.number_proc), 
                                    suffix="cleaned", inverse=True, angle=-90, general=True)
        return cleaned_image

    def _rotate_image(self, image, angle):
        self.number_proc += 1
        image_rotated = ImageUtils.rotate_image(image, angle=angle)
        self.out_module.save_image(image_rotated, prefix=str(self.number_proc),
                                        suffix="rotated", inverse=True)
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
            raise ValueError("That option is not available")
        self.number_proc += 1
        self.out_module.save_image(image_contours, prefix=str(self.number_proc),
                                suffix="contours", inverse=True)

        return image_contours, lines
    
    def _extract_characters(self, line, line_count, extract_character_args):
        self.out_module.save_image(line, folder="line", prefix="",
                            suffix="image_line-{}".format(line_count))
        self.out_module.image_objects["lines"]["line-{}".format(line_count)] = {}
        self.out_module.image_objects["lines"]["line-{}".format(line_count)]["base"] = line
        image_color, characters, params_vproj = Hoof_VThresh.vertical_projection_segmentation(line, **extract_character_args)
        self.out_module.save_projection_image(image_color, prefix="image_vertical_projection_line_{}".format(line_count),
                        projection=params_vproj["projection"], minima=params_vproj["minima"],
                        maxima=params_vproj["maxima"], line_count=line_count, inverse=True)
        self.out_module.image_objects["lines"]["line-{}".format(line_count)]["line_div"] = image_color
        self.out_module.image_objects["lines"]["line-{}".format(line_count)]["line_div_params"] = params_vproj
        self.out_module.image_objects["lines"]["line-{}".format(line_count)]["characters"] = {}
        char_count = 0
        for idx,char in enumerate(characters):
            self.out_module.image_objects["lines"]["line-{}".format(line_count)]["characters"]["character-{}".format(idx)] = {}
            self.out_module.image_objects["lines"]["line-{}".format(line_count)]["characters"]["character-{}".format(idx)]["original"] = char
            self.out_module.save_image(char, folder="character",prefix="", 
                                suffix="image_line-{}_character-{}".format(line_count, char_count), inverse=self.inverted_img)
            
            char_count += 1
        return characters
    
    def _translate_characters(self, characters, straw, notransform_input, line_count, infer_model_args):

        dataloader = straw.load_data(characters, notransform_input)
        results = straw.infer_model(dataloader, **infer_model_args)
        outputs = []
        for idx, (k, val) in enumerate(results.items()):
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
            outputs.append(val["output"])
        #self.out_module.plot_allsentence_images(line_num=line_count, amount_charact=len(results))
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
            extra_metadaa = tongue.charcterize_sentence(sentence["sentence"])
            sentence.update(extra_metadaa)
            sentence["probability"] = Tongue.perplexity_to_score_log(sentence["perplexity"])
        return sentences_demo, quantiled_sentences, tongue.character
    
    @staticmethod
    def binary_with_probability(x):
        return 1 if random.random() < x else 0


    def main(self, image_path, config, random_pick=False,
                on_event=None, should_cancel=None):
        """Run the full pipeline on one image.

        ``config`` is a :class:`luvia.config.PipelineConfig` carrying all the
        per-stage settings. ``on_event`` and ``should_cancel`` default to
        no-ops so the CLI dispatcher and tests can call ``main(path, config)``
        without extra wiring.
        """
        on_event = on_event or (lambda name, payload=None: None)
        should_cancel = should_cancel or (lambda: False)
        def _check():
            if should_cancel():
                raise CancelledError("Pipeline cancelled by user")
        self.out_module = OutUtils(base_folder=self.out_folder, mode=self.mode, filename=os.path.basename(image_path))
        on_event("started", {"image_path": image_path, "mode": "main",
                              "output_folder": str(self.out_module.output_folder)})
        print("======================= ANALYZING STREET IMAGE =======================")
        self.image = self.first_step(image_path=image_path, invert=self.inverted_img)
        on_event("image_loaded", {})
        _check()

        if not config.clean_mode:
            cleaned_image = self.image
        else:
            print("======================= CLEANING STREET IMAGE =======================")
            cleaned_image = self._clean_image(
                image=self.image,
                clean_image=config.clean_mode,
                clean_args=config.clean.model_dump() if config.clean else {},
            )
        on_event("cleaned", {})
        _check()
        print("======================= ROTATING STREET IMAGE =======================")
        image_rotated = self._rotate_image(image=cleaned_image, angle=config.rotate_img)
        on_event("rotated", {})
        _check()
        print("======================= EXTRACTING SMEDT SHORTHAND SENTENCES =======================")
        image_contours, lines = self._extract_sentences(
            image_rotated=image_rotated,
            extract_sentences=config.hoofh_mode,
            extract_lines_args=config.hoof_v.model_dump(),
        )
        on_event("lines_extracted", {"count": len(lines)})
        _check()
        self.out_module.plot_alltransformations()
        straw = Straw()
        straw.load_model(config.straw.weights)
        # Pre-compute the per-character inference kwargs (everything except
        # the loader controls). model_dump() produces a fresh dict per main()
        # invocation; no mutation of caller state.
        infer_kwargs = config.straw.model_dump(exclude={"weights", "notransform_input"})
        notransform_input = config.straw.notransform_input
        tongue_cfg = config.tongue
        sentences_demo = []
        if random_pick:
            random.shuffle(lines)
        character_chosen = ""
        print("======================= TRANSLATING SMEDT SHORTHAND SENTENCES =======================")
        for line_count, line in tqdm(enumerate(lines)):
            _check()
            on_event("line_started", {"line": line_count, "total": len(lines)})
            print("======================= TRANSLATING SMEDT SHORTHAND SENTENCE NUMBER {} =======================".format(line_count+1))
            characters = self._extract_characters(line, line_count, config.hoof_h.model_dump())
            if random_pick and LUVIA.binary_with_probability(0.4):
                if len(characters) == 1:
                    continue
            outputs = self._translate_characters(characters, straw=straw, notransform_input=notransform_input,
                                                line_count=line_count, infer_model_args=infer_kwargs)
            if len(outputs) == 0:
                print("Sentence {} doesnt have any character".format(line_count))
                continue
            print("======================= MORPHING SMEDT SHORTHAND SENTENCE NUMBER {} =======================".format(line_count+1))
            candidate_sentences, sentences_info, character_chosen = self._morph_sentence(
                outputs=outputs,
                dictionary=tongue_cfg.dictionary,
                character=tongue_cfg.character,
                corrected_k=tongue_cfg.corrected_k,
                sel_sentence=tongue_cfg.sel_sentence,
                quantile=tongue_cfg.quantile,
                final_sentences=tongue_cfg.final_sentences,
            )
            sentences_demo.append(sentences_info)
            on_event("line_morphed", {"line": line_count,
                                       "sentence": sentences_info[0]["sentence"] if sentences_info else ""})
            for k, word in enumerate(sentences_info[0]["sentence"].split(" ")):
                key = "line-{}_character-{}".format(line_count, k)
                try:
                    self.out_module.plot_allchar_images(suffix=key, line_count=line_count,
                                                    word=word, sentence_info=sentences_info[0])
                except KeyError:
                    break
            if random_pick:
                break
        location = os.path.basename(image_path).split(".")[0]
        self.out_module.create_pdftranslation(user=self.username, character=character_chosen,
                                              sentences_data=sentences_demo, location=location)
        on_event("finished", {"output_folder": str(self.out_module.output_folder),
                               "lines_processed": len(sentences_demo)})
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

    
    def horde(self, folder_streets, config,
                limit_loops=False, max_runs=10, num_workers=1,
                on_event=None, should_cancel=None):
        """Run main() over a random folder of street images in a loop.

        ``config`` is a :class:`luvia.config.PipelineConfig`; this method
        randomises ``rotate_img`` per iteration and forces ``clean_mode=OTSA``
        and ``hoofh_mode=cca`` (the established horde defaults), leaving
        everything else as the caller specified.
        """
        on_event = on_event or (lambda name, payload=None: None)
        should_cancel = should_cancel or (lambda: False)
        self.out_module = OutUtils(base_folder=self.out_folder, mode=self.mode, filename="LOOP")
        on_event("horde_started", {"output_folder": str(self.out_module.output_folder),
                                    "folder_streets": folder_streets,
                                    "num_workers": num_workers})
        dict_files = self._getstreets(folder_streets=folder_streets)
        if num_workers > 1:
            return self._horde_parallel(
                dict_files=dict_files, config=config,
                limit_loops=limit_loops, num_workers=num_workers,
                on_event=on_event, should_cancel=should_cancel)
        count_runs = 0
        rotate_angles = np.arange(-180, 190, 10)
        json_path = "{}/LUVIA_history.jsonl".format(self.out_module.output_folder)
        runs_folder = []
        while True:
            if should_cancel():
                on_event("horde_cancelled", {"count": count_runs})
                break
            file_key = random.choice(list(dict_files.keys()))
            file_path = dict_files[file_key]
            angle = int(random.choice(rotate_angles))
            on_event("horde_iteration_started",
                     {"count": count_runs + 1, "file": file_path, "angle": angle})

            iter_config = config.model_copy(update={
                "rotate_img": angle,
                "clean_mode": "OTSA",
                "hoofh_mode": "cca",
            })

            main_instance = self.__class__(inverted_img=self.inverted_img,
                                            out_folder=self.out_module.output_folder,
                                            user=self.username,
                                            mode="main")
            try:
                sentences, out_folder = main_instance.main(
                    image_path=file_path, config=iter_config,
                    random_pick=True,
                    on_event=on_event, should_cancel=should_cancel)
            except CancelledError:
                on_event("horde_cancelled", {"count": count_runs})
                break
            except TypeError:
                on_event("horde_iteration_failed",
                         {"count": count_runs + 1, "reason": "TypeError"})
                continue
            shutil.copy("{}/image-transformation.jpg".format(out_folder),
                        "{}/images/image-transformation.jpg".format(self.out_module.output_folder))
            runs_folder.append(out_folder)
            sentence_num = 0
            entry ={"sentence0": {
                        "sentence": sentences[0][0]["sentence"],
                        "probability":float(sentences[0][0]["probability"])},
                    "sentence1": {
                        "sentence": sentences[0][1]["sentence"],
                        "probability":float(sentences[0][1]["probability"])},
                    "sentence2": {
                        "sentence": sentences[0][2]["sentence"],
                        "probability":float(sentences[0][2]["probability"])},
                    "location": "{}--56,24".format(file_key),
                    "image": ["{}/images/line_images/_image_line-{}.jpg".format(out_folder, sentence_num),
                              "{}/images/3_contours.jpg".format(out_folder, sentence_num)],
                    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "id": os.path.basename(out_folder)
                    }
            self._write_jsonfile(json_path=json_path, new_entry=entry)
            on_event("horde_entry_written", {"count": count_runs + 1, "entry": entry})
            if False and len(runs_folder) >=max_runs:
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
                    on_event("horde_finished", {"count": count_runs})
                    break

    def _horde_parallel(self, dict_files, config,
                         limit_loops, num_workers, on_event, should_cancel):
        """Parallel horde loop using a ProcessPoolExecutor.

        Each worker process loads its own copy of the models on first iteration
        and reuses them via the module-scope cache for subsequent iterations,
        so amortised per-iteration cost stays similar to the sequential path.
        on_event / should_cancel are owned by this process; workers return
        plain dicts that we surface as events.
        """
        import concurrent.futures
        import multiprocessing

        rotate_angles = np.arange(-180, 190, 10)
        json_path = "{}/LUVIA_history.jsonl".format(self.out_module.output_folder)

        init_kwargs = {
            "inverted_img": self.inverted_img,
            "out_folder": str(self.out_module.output_folder),
            "user": self.username,
        }

        count_completed = 0
        next_iter = [1]
        pending = {}
        cancelled = [False]

        def can_submit_more():
            return not should_cancel() and (not limit_loops or next_iter[0] <= limit_loops)

        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers,
                                                     mp_context=mp_ctx) as pool:
            def submit_next():
                file_key = random.choice(list(dict_files.keys()))
                file_path = dict_files[file_key]
                angle = int(random.choice(rotate_angles))
                iter_config = config.model_copy(update={
                    "rotate_img": angle,
                    "clean_mode": "OTSA",
                    "hoofh_mode": "cca",
                })
                fut = pool.submit(_horde_iteration_worker, file_path,
                                  file_key, init_kwargs, iter_config)
                pending[fut] = (file_key, file_path, angle, next_iter[0])
                on_event("horde_iteration_started",
                         {"count": next_iter[0], "file": file_path, "angle": angle})
                next_iter[0] += 1

            for _ in range(num_workers):
                if not can_submit_more():
                    break
                submit_next()

            while pending:
                if should_cancel() and not cancelled[0]:
                    on_event("horde_cancelled", {"count": count_completed})
                    cancelled[0] = True

                done, _ = concurrent.futures.wait(
                    pending.keys(), timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED)

                for fut in done:
                    file_key, file_path, angle, iter_num = pending.pop(fut)
                    try:
                        result = fut.result()
                    except TypeError:
                        on_event("horde_iteration_failed",
                                 {"count": iter_num, "reason": "TypeError"})
                        if can_submit_more():
                            submit_next()
                        continue

                    try:
                        shutil.copy(
                            "{}/image-transformation.jpg".format(result["out_folder"]),
                            "{}/images/image-transformation.jpg".format(
                                self.out_module.output_folder))
                    except FileNotFoundError:
                        pass

                    sentences = result["sentences"]
                    out_folder = result["out_folder"]
                    entry = {
                        "sentence0": {
                            "sentence": sentences[0][0]["sentence"],
                            "probability": float(sentences[0][0]["probability"])},
                        "sentence1": {
                            "sentence": sentences[0][1]["sentence"],
                            "probability": float(sentences[0][1]["probability"])},
                        "sentence2": {
                            "sentence": sentences[0][2]["sentence"],
                            "probability": float(sentences[0][2]["probability"])},
                        "location": "{}--56,24".format(file_key),
                        "image": [
                            "{}/images/line_images/_image_line-0.jpg".format(out_folder),
                            "{}/images/3_contours.jpg".format(out_folder)],
                        "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        "id": os.path.basename(out_folder),
                    }
                    self._write_jsonfile(json_path=json_path, new_entry=entry)
                    on_event("horde_entry_written",
                             {"count": iter_num, "entry": entry})
                    count_completed += 1

                    if can_submit_more():
                        submit_next()

        if not cancelled[0]:
            on_event("horde_finished", {"count": count_completed})
            



def _horde_iteration_worker(file_path, file_key, init_kwargs, iter_config):
    """Module-level worker for parallel horde -- must be picklable.

    Runs one full main() iteration in a subprocess. ``iter_config`` is a
    :class:`luvia.config.PipelineConfig` already customised by the parent
    (rotate_img, clean_mode, hoofh_mode). on_event / should_cancel are
    intentionally NOT passed across the process boundary; the parent emits
    events based on the returned result / raised exception.
    """
    main_instance = LUVIA(mode="main", **init_kwargs)
    sentences, out_folder = main_instance.main(
        image_path=file_path,
        config=iter_config,
        random_pick=True,
    )
    return {
        "sentences": sentences,
        "out_folder": str(out_folder),
        "file_key": file_key,
        "file_path": file_path,
        "angle": iter_config.rotate_img,
    }


def run_from_args(largs, on_event=None, should_cancel=None):
    """Dispatch a parsed argparse Namespace to the right LUVIA method.

    Shared by the CLI entry point and the GUI's PipelineWorker. Builds a
    typed :class:`luvia.config.PipelineConfig` from ``largs`` and hands it
    to ``LUVIA.main`` or ``LUVIA.horde``.
    """
    config = PipelineConfig.from_namespace(largs)
    l = LUVIA(inverted_img=largs.inverted_image, out_folder=largs.output,
              user=largs.user, mode=largs.command)
    if largs.command == "main":
        l.main(image_path=largs.input, config=config,
                on_event=on_event, should_cancel=should_cancel)
    elif largs.command == "horde":
        l.horde(folder_streets=largs.folder_streets, config=config,
                limit_loops=False, num_workers=largs.num_workers,
                on_event=on_event, should_cancel=should_cancel)


def main():
    largs = LUVIAargs.main()
    run_from_args(largs)
    print("======================= LUVIA RUN SUCCESSFULLY =======================")
    print("======================================================================")
if __name__== "__main__":
    main()
    

            


