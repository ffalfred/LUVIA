import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import difflib
import os
import spacy

from luvia.straw.model.model import ImageToText
from luvia.straw.actions import NeuralActions
from luvia.straw.utils.data_utils import Shorthand_Dataset, Shorthand_Data

class Straw:

    vocab_dict = {'<PAD>': 0, '<START>': 1, '<END>': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7,
                    'f': 8, 'g': 9, 'h': 10, 'i': 11, 'j': 12, 'k': 13, 'l': 14, 'm': 15, 'n': 16,
                    'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21, 't': 22, 'u': 23, 'v': 24,
                    'w': 25, 'x': 26, 'y': 27, 'z': 28}
    maxlen_word = 21


    def __init__(self, db_words=False, vocab_dict=None, device="cuda"):

        if vocab_dict is None:
            self.vocab_dict = Straw.vocab_dict
        else:
            self.vocab_dict = vocab_dict
        self.model = ImageToText(vocab_size=len(self.vocab_dict))
        self.model = self.model.to(device)
        self.device = device
        self.valid_words = []
        if db_words:
            for filename in os.listdir(db_words):
                self.valid_words.append(filename.replace(".png", ""))


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

    def load_data(self, files):
        data = Shorthand_Data(files)
        data_loader = DataLoader(data, batch_size=64, num_workers=8, collate_fn=Shorthand_Data.collate)
        return data_loader

    def infer_model(self, data_loader, mode="vanilla", length_norm=True, beam_width=3, num_groups=3,
                    diversity_strength=0.5, top_k=0, top_p=0.9, temperature=1.0, k=1):

        vocab_inv_dict = {v: k for k, v in Straw.vocab_dict.items()}
        results = {}

        for batch_idx, (images, paths) in tqdm(enumerate(dataloader)):
            images = images.to(self.device)
            for i in range(len(images)):
                results[paths[i]] = []

                output = self.model.infer(image=images[i], start_token=self.vocab_dict['<START>'], end_token=self.vocab_dict['<END>'],
                                    beam_width=beam_width, max_len=Straw.maxlen_word, length_norm=length_norm, mode=mode,
                                    num_groups=num_groups, diversity_strength=diversity_strength, top_k=top_k, top_p=top_p, temperature=temperature,
                                    k=k)
                output = output
                for out in output:
                    out = out.cpu()
                    decoded = ''.join([vocab_inv_dict[idx.item()] for idx in out[:-1]])
                    results[paths[i]].append(decoded)
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
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
    

if __name__== "__main__":
    straw = Straw(db_words="../../../../data/gregg_definitive/")

    train_loader = straw.load_dataset(folder="../../../../data/gregg_definitive/", metadata="../utils/greggs_metadata.tsv",
                                        subset='train', augmentation=True,
                                        freqw_file="../utils/general_POS_freq_speak.txt")
    val_loader = straw.load_dataset(folder="../../../../data/gregg_definitive/", subset='val', metadata="../utils/greggs_metadata.tsv",)

    straw.train_model(train_loader=train_loader, val_loader=val_loader, epochs=60)
    straw.save_model(path="./weights/weights_speakcorpus_e60.pt")
    exit()
    straw.load_model(path="./straw/weights/try1.pt")
    dataloader = straw.load_data(["../../../data/gregg_definitive/incentive.png", "../../../data/gregg_definitive/seemingly.png",
                                    "../../../data/gregg_definitive/miner.png"])
    results = straw.infer_model(dataloader, "diverse_beam", k=3)
    print(results)
    results_filtered = {}
    for k, val in results.items():
        filtered_words = straw.distance_annotations(val, n=2)
        results_filtered[k] = filtered_words
    print(results_filtered)
            