import pandas as pd
import numpy as np
import os

from skimage.transform import resize
import skimage.io as img_io

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

import albumentations as A

class Shorthand_Data(Dataset):

    def __init__(self, files, fixed_size=(48, 56)):

        self.files = files
        self.fixed_size = fixed_size

    def std_image(self, img_path):
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
 #       img = Shorthand_Dataset.load_image(img_path)
        img = Shorthand_Dataset.preprocess(img, (fheight, fwidth))
        img = torch.Tensor(img).float().unsqueeze(0)
        return img

    def __getitem__(self, index):
        if isinstance(self.files[index], str):
            img_path = os.path.abspath(self.files[index])
            img = Shorthand_Dataset.load_image(img_path)
        else:
            img = self.files[index]
        img = self.std_image(img)
        return img, img_path

    def __len__(self):
        return len(self.files)

    @staticmethod
    def collate(batch):
        (_img, path) = zip(*batch)
        img = []
        for i in _img:
            img.append(i)
        img = torch.stack(img,0)
        paths = []
        for p in path:
            paths.append(p)
        return img, paths



class Shorthand_Dataset(Dataset):

    def __init__(self, basefolder: str = 'IAM/', metadata:str = None, subset: str = 'train', fixed_size=(48, 56), 
                    transforms: list = None, character_classes: list = None, max_length = 21, rnd_subset = False):

        self.basefolder = basefolder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.character_classes = character_classes
        self.max_length = max_length

        data = self.create_data()
        
        if character_classes is None:
          self.character_classes = self.create_classes(data)
        else:
          self.character_classes = character_classes
        
        self.char_to_num, self.num_to_char = self.create_dicts()

        self.data = pd.DataFrame(data,columns=["img_path", "word"])

        if metadata is not None:
            metadata = pd.read_csv(metadata, sep="\t")
            self.data = self.data.merge(metadata, on="word")
        self.data  = self.data.sample(frac=1).reset_index(drop=True)

        if rnd_subset:
          self.data = self.data.sample(frac=0.1)

    def create_data(self):
        data = []
        for filename in os.listdir(self.basefolder):
          if not filename.endswith('.png'):
            continue
          transcr = filename.replace('.png', '')
          data += [(os.path.join(self.basefolder, filename), transcr)]
        return data

    def create_dicts(self):
        char_to_num = {c:(i) for i,c in enumerate(self.character_classes)}
        num_to_char = {(i):c for i,c in enumerate(self.character_classes)}
        return char_to_num, num_to_char

    def create_weights(self, file_freq=None):
        cols = ["Noun","Verb","Adjective","Adverb","Preposition","Foreign","Pronoun","Interjection",
                "Conjunction","Determiner","Numeral","Particle","Other","Existential"]
        if file_freq is not None:
            freq_df = pd.read_csv(file_freq, sep="\t")
            self.data["Frequency_POS"] = 0
            self.data["Count_POS"] = self.data[cols].sum(axis=1)
            for c in cols:
                if c != "Particle":
                    frequency = (self.data[c]*float(freq_df.loc[freq_df["General_POS"]==c, "Perc"].values[0]))/sum(self.data[c])
                else:
                    frequency = (self.data[c]*float(sorted(freq_df["Perc"])[0]))/sum(self.data[c])       
                self.data["Frequency_POS"]+=frequency
            self.data["Frequency_POS"] /= self.data["Count_POS"]
            return self.data["Frequency_POS"]
        else:
            return [1]*len(self.data)

    def word_to_vec(self, word):
        return np.array([self.char_to_num[c] for c in word])

    def vec_to_word(self, vec):
        return ''.join([self.num_to_char[c] for c in vec])

    def create_classes(self, data, classes=None):
        if classes is None:
            res = set()
            for _,transcr in data:
                res.update(list(transcr))
            res = sorted(list(res))
            classes = ["<PAD>", "<START>","<END>"] + list(res)
        return classes

    def get_image(self, img_path):
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
        img = Shorthand_Dataset.load_image(img_path)

#        if self.subset == 'train':
 #           nwidth = int(np.random.uniform(.75, 1.25) * img.shape[1])
  #          nheight = int((np.random.uniform(.9, 1.1) * img.shape[0] / img.shape[1]) * nwidth)
   #         img = resize(image=img, output_shape=(nheight, nwidth)).astype(np.float32)
        img = Shorthand_Dataset.preprocess(img, (fheight, fwidth))

        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        img = torch.Tensor(img).float().unsqueeze(0)
        return img


    def __getitem__(self, index):
        img_path = self.data.iloc[index]["img_path"]
        img = self.get_image(img_path)

        transcr = self.data.iloc[index]["word"]
        spl_transcr = ["<START>"]+list(transcr)
        spl_transcr.append('<END>')
        int_transcr = self.word_to_vec(spl_transcr)
        type_str = self.data.iloc[index]["GEN_POSTAGS"]
        return img, transcr, spl_transcr, torch.from_numpy(int_transcr), len(spl_transcr)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def pad_collate(batch):
        (_img, _transcr, _spl_transcr, _int_transcr, len_transcr) = zip(*batch)
        img = []
        for i in _img:
            img.append(i)
        img = torch.stack(img,0)
        transcr = []
        for i in _transcr:
            transcr.append(i)
        len_transcr = np.array(len_transcr)
        spl_transcr = []
        for i in _spl_transcr:
            spl_transcr.append(i+["<PAD>"]*(64-len(i)))
        int_transcr = pad_sequence(_int_transcr, batch_first=True, padding_value=0)

        return img, transcr, spl_transcr, int_transcr, torch.Tensor(len_transcr)

    @staticmethod
    def load_image(image_path):

        # read the image
        image = img_io.imread(image_path,as_gray=True)

        # convert to grayscale skimage
        if len(image.shape) == 3:
            image = img_color.rgb2gray(image)
        # normalize the image
        image = 1 - image

        return image

    @staticmethod
    def preprocess(img, input_size, border_size=8):
        h_target, w_target = input_size
        n_height = min(h_target - 2 * border_size, img.shape[0])
        scale = n_height / img.shape[0]
        n_width = min(w_target - 2 * border_size, int(scale * img.shape[1]))
        # resize the image
        img = resize(image=img, output_shape=(n_height, n_width)).astype(np.float32)
        # right pad image to input_size
        img  = np.pad(img, ((border_size, h_target - n_height - border_size), (border_size, w_target - n_width - border_size),),
                        mode='constant', constant_values=0)
        return img

    @staticmethod
    def augmentation_functions(geometric=True, perspective=False, distortions=True, erosion=True, col_invertion=False,
                                col_augmentation=True, col_contrast=True):
        aug_lst = []
        # geometric augmentation
        if geometric:
            geom_c = A.Affine(rotate=(-1, 1), shear={'x':(-30, 30), 'y' : (-5, 5)}, scale=(0.6, 1.2),
                                translate_percent=0.02, mode=1, p=0.5)
            aug_lst.append(geom_c)
        # perspective transform
        if perspective:
            persp_c = A.Perspective(scale=(0.05, 0.1), p=0.5)
            aug_lst.append(persp_c)
        # distortions
        if distortions:
            dist_c = A.OneOf([A.GridDistortion(distort_limit=(-.1, .1), p=0.5),
                                A.ElasticTransform(alpha=60, sigma=20, alpha_affine=0.5, p=0.5),], p=0.5)
            aug_lst.append(dist_c)
        # erosion & dilation
        if erosion:
            ero_c = A.OneOf([A.Morphological(p=0.5, scale=3, operation='dilation'),
                            A.Morphological(p=0.5, scale=3, operation='erosion'),], p=0.5)
            aug_lst.append(ero_c)
        # color invertion - negative
        if col_invertion:
            colinv = A.InvertImg(p=0.5)
            aug_lst.append(colinv)
        # color augmentation - only grayscale images
        if col_augmentation:
            colaug = A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2)
            aug_lst.append(colaug)
        # color contrast
        if col_contrast:
            colcon = A.RandomGamma(p=0.5, gamma_limit=(80, 120))
            aug_lst.append(colcon)
        return A.Compose(aug_lst)

if __name__== "__main__":
    aug_transforms = Shorthand_Dataset.augmentation_functions()
    train_create = Shorthand_Dataset(basefolder="../../../../../data/gregg_definitive/", metadata="../../utils/greggs_metadata.tsv",
                                    subset='train', max_length = 21, rnd_subset = False, transforms=aug_transforms)
    weights = train_create.create_weights("../../utils/general_POS_freq_speak.txt")

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_create, batch_size=64, shuffle=False, num_workers=8,
                                    collate_fn=Shorthand_Dataset.pad_collate, sampler=sampler)
    for t in train_loader:
        print(pd.Series(t[-1]).value_counts())
        break
    exit()
    datal = Shorthand_Data(["../../../../../data/gregg_definitive/incentive.png", "../../../../data/gregg_definitive/seemingly.png",
                                    "../../../../data/gregg_definitive/miner.png"])
    dataloader = DataLoader(datal, batch_size=64, num_workers=8, collate_fn=Shorthand_Data.collate)
    for da in dataloader:
        print(da[0].shape)
    
