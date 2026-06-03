import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import difflib
import os
import pandas as pd
import spacy
from pathlib import Path
import random

from luvia.straw.model.model import ImageToText
from luvia.straw.actions import NeuralActions
from luvia.straw.utils.data_utils import Shorthand_Dataset, Shorthand_Data


from luvia.straw.model.encoder import GuidedBackpropModel


class Straw:

    vocab_dict = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7,
                    'f': 8, 'g': 9, 'h': 10, 'i': 11, 'j': 12, 'k': 13, 'l': 14, 'm': 15, 'n': 16,
                    'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21, 't': 22, 'u': 23, 'v': 24,
                    'w': 25, 'x': 26, 'y': 27, 'z': 28}
    maxlen_word = 21

    _weights_dir = "{}/../data/weights".format(os.path.dirname(os.path.abspath(__file__)))
    weights_model = {
        "speak": "{}/weights2_speakcorpus_e60.pt".format(_weights_dir),
        "all": "{}/weights_allcorpus_e60.pt".format(_weights_dir),
        "conversation": "{}/weights_conversationcorpus_e60.pt".format(_weights_dir),
        "imaginative": "{}/weights_imaginativecorpus_e60.pt".format(_weights_dir),
        "informative": "{}/weights_informativecorpus_e60.pt".format(_weights_dir),
        }


    def __init__(self, vocab_dict=None):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if vocab_dict is None:
            self.vocab_dict = Straw.vocab_dict
        else:
            self.vocab_dict = vocab_dict
        self.model = ImageToText(vocab_size=len(self.vocab_dict))
        self.model = self.model.to(device)
        self.device = device


    def load_dataset(self, folder, subset, metadata, num_workers=8, batch_size=64, augmentation=False, freqw_file=None):
        if augmentation:
            aug_transforms = Shorthand_Dataset.augmentation_functions()
        else:
            aug_transforms = None
        if subset == "train" or subset == "infer":
            rnd_subset = False
        else:
            rnd_subset = True
        dataset = Shorthand_Dataset(basefolder=folder, subset=subset, metadata=metadata, max_length=Straw.maxlen_word,
                                        rnd_subset=rnd_subset, transforms=aug_transforms)
        weights = dataset.create_weights(freqw_file)

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        if subset == "infer":
            shuffle = False
        else:
            shuffle = True
        set_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                    collate_fn=Shorthand_Dataset.pad_collate, sampler=sampler)
        return set_loader

    def load_data(self, files, transform=True):
        data = Shorthand_Data(files, transforms=transform)
        data_loader = DataLoader(data, batch_size=64, num_workers=8, collate_fn=Shorthand_Data.collate)
        return data_loader

    @staticmethod
    def occlusion_sensitivity(image_tensor, model, patch_size=8):
        # Sensitivity is forward-only. Switch to eval mode so BatchNorm uses
        # running statistics (not per-sample), which lets us score every
        # occluded patch in one batched forward pass instead of one forward
        # per patch (~42x fewer forwards for a 48x56 image with patch_size=8).
        image_tensor = image_tensor.clone()
        _, _, H, W = image_tensor.shape

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                base_output, _, _ = model(image_tensor)
                base_score = base_output[0].sum().item()

                patches = []
                coords = []
                for i in range(0, H, patch_size):
                    for j in range(0, W, patch_size):
                        occluded = image_tensor.clone()
                        occluded[0, 0, i:i+patch_size, j:j+patch_size] = 0
                        patches.append(occluded)
                        coords.append((i, j))

                batch = torch.cat(patches, dim=0)
                outputs, _, _ = model(batch)
                scores = outputs.reshape(outputs.size(0), -1).sum(dim=1)
        finally:
            if was_training:
                model.train()

        sensitivity_map = torch.zeros(H, W)
        for idx, (i, j) in enumerate(coords):
            sensitivity_map[i:i+patch_size, j:j+patch_size] = base_score - scores[idx].item()
        return sensitivity_map


    @staticmethod
    def get_saliency_map(input_tensor, model):
        """Per-sample saliency for a batched input.

        input_tensor: (N, 1, H, W) -- pass a whole batch of characters; the
        encoder treats samples independently so one forward + one backward
        yields per-sample gradients. Returns: (N, H, W).

        Model is switched to eval() so BatchNorm uses running statistics,
        matching the occlusion_sensitivity batching convention.
        """
        was_training = model.training
        model.eval()
        try:
            input_tensor = input_tensor.detach().clone().requires_grad_()
            output, _, _ = model(input_tensor)
            output.sum().backward()
            return input_tensor.grad.data.abs().squeeze(1).cpu()
        finally:
            if was_training:
                model.train()

    @staticmethod
    def getguidedbackprop(input_tensor, model):
        """Per-sample guided backprop for a batched input.

        Same batching/eval convention as get_saliency_map.
        input_tensor: (N, 1, H, W); returns: (N, H, W).
        """
        was_training = model.training
        model.eval()
        try:
            input_tensor = input_tensor.detach().clone().requires_grad_()
            gb_model = GuidedBackpropModel(model)
            output = gb_model(input_tensor)
            output.sum().backward()
            return input_tensor.grad.data.squeeze(1).cpu()
        finally:
            if was_training:
                model.train()


    def infer_model(self, data_loader, infer_mode="vanilla", length_norm=True, beam_width=3, num_groups=3,
                    diversity_strength=0.5, top_k=0, top_p=0.9, temperature=1.0, k=1):

        vocab_inv_dict = {v: k for k, v in Straw.vocab_dict.items()}
        results = {}

        for images, paths in data_loader:
            images = images.to(self.device)
            n = len(images)

            # Saliency + guided backprop are batched across the whole
            # dataloader batch -- one forward + backward each, instead of
            # N. Sensitivity stays per-image because batching across images
            # would multiply the already-batched patch tensor.
            saliencies = Straw.get_saliency_map(images, self.model.encoder)
            gb_grads = Straw.getguidedbackprop(images, self.model.encoder)

            for i in tqdm(range(n)):
                results[paths[i]] = {"output": []}
                output, act1, act2 = self.model.infer(
                    image=images[i],
                    start_token=self.vocab_dict['<START>'],
                    end_token=self.vocab_dict['<END>'],
                    beam_width=beam_width, max_len=Straw.maxlen_word,
                    length_norm=length_norm, mode=infer_mode,
                    num_groups=num_groups, diversity_strength=diversity_strength,
                    top_k=top_k, top_p=top_p, temperature=temperature, k=k)
                results[paths[i]]["act1"] = act1.cpu().detach().numpy()
                results[paths[i]]["act2"] = act2.cpu().detach().numpy()
                results[paths[i]]["conv1"] = self.model.encoder.conv1.weight
                results[paths[i]]["conv2"] = self.model.encoder.conv2.weight
                results[paths[i]]["saliency"] = saliencies[i]
                results[paths[i]]["sensitivity"] = Straw.occlusion_sensitivity(
                    images[i].unsqueeze(0), self.model.encoder)
                results[paths[i]]["gb_grad"] = gb_grads[i]
                for out in output:
                    out = out.cpu()
                    decoded = ''.join([vocab_inv_dict[idx.item()] for idx in out[:-1]])
                    results[paths[i]]["output"].append(decoded)
        return results

    def distance_annotations(self, candidates, n=1):
        if len(self.valid_words) == 0:
            raise KeyError("No valid words availables")
        closest_matches = {}
        for word in candidates:
            # Use difflib to find the closest match
            match = difflib.get_close_matches(word, self.valid_words, n=n, cutoff=0.0)
            if match:
                closest_matches[word] = match
            else:
                closest_matches[word] = None
        return closest_matches
      

    def train_model(self, train_loader, val_loader, epochs=15):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        loss_train, loss_val = NeuralActions.train_and_validate(model=self.model,
                                                    train_loader=train_loader, val_loader=val_loader,
                                                    optimizer=optimizer, vocab=self.vocab_dict,
                                                    pad_idx=self.vocab_dict["<PAD>"], device=self.device,
                                                    num_epochs=epochs, max_len=Straw.maxlen_word)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    

    def load_model(self, model_weights=False):
        if os.path.isfile(model_weights):
            path_weight = model_weights
        else:
            if model_weights == "random":
                path_weight = random.choice(list(Straw.weights_model.items()))[1]
            else:
                path_weight = Straw.weights_model[model_weights]
        state = torch.load(path_weight, map_location=self.device, weights_only=True)
        # Weights are stored as float16 on disk to keep the bundle small;
        # upcast to float32 for inference (BatchNorm and the rest of the CNN
        # run in float32 on CPU and GPU).
        state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        self.model.load_state_dict(state)
    

if __name__== "__main__":
    straw = Straw()
#    exit()

    #train_loader = straw.load_dataset(folder="../../../../data/gregg_definitive/", metadata="../data/greggs_metadata.tsv",
    #                                    subset='train', augmentation=True,
   #                                     freqw_file="../../../data/general_POS/general_POS_freq_speak.txt")
 #   val_loader = straw.load_dataset(folder="../../../../data/gregg_definitive/", subset='val', metadata="../data/greggs_metadata.tsv",)
#
  #  straw.train_model(train_loader=train_loader, val_loader=val_loader, epochs=60)
   # straw.save_model(path="../data/weights/weights2_speakcorpus_e60.pt")
    #exit()
    straw.load_model(path="./data/weights/try1.pt")
    dataloader = straw.load_data(["../../../data/gregg_definitive/incentive.png", "../../../data/gregg_definitive/seemingly.png",
                                    "../../../data/gregg_definitive/miner.png"])
    results = straw.infer_model(dataloader, "diverse_beam", k=3)
    print(results)
    results_filtered = {}
    for k, val in results.items():
        filtered_words = straw.distance_annotations(val, n=2)
        results_filtered[k] = filtered_words
    print(results_filtered)
            
