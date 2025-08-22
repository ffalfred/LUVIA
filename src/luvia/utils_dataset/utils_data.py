import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
import wiktionaryparser
#from wiktionaryparser import WiktionaryParser
from tqdm import tqdm
from collections import defaultdict
from itertools import chain




from nltk.corpus import wordnet as wn
from nltk.corpus import brown

# Download required resources
nltk.download('wordnet')
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')



class PolishDB():

    def __init__(self, brown_to_gen, pos_tags, freqs_tags):

        self.pos_tags = pd.read_csv(pos_tags, sep="\t")
        self.freqs_tags = pd.read_csv(freqs_tags, sep="\t")
        self.postags_dict = PolishDB.build_brown_pos_dict()
        self.brown_to_general = pd.read_csv(brown_to_gen)
    
    def makeDF(self, basefolder, dbout):
        data = []
        for filename in os.listdir(basefolder):
          if not filename.endswith('.png'):
            continue
          transcr = filename.replace('.png', '')
          data += [(filename, transcr)]
        datadf = pd.DataFrame(data, columns=["file_path", "word"])   
        analysis_words = self.analyze_words(datadf["word"].tolist())
        analysis_df = pd.DataFrame(analysis_words)
        list_gen_pos = ["Noun", "Verb", "Adjective", "Adverb", "Preposition", "Foreign", "Pronoun", "Interjection", "Conjunction",
                        "Determiner", "Numeral", "Particle", "Other", "Existential"] 
        for l in list_gen_pos:
            analysis_df[l] = 0
            analysis_df.loc[analysis_df["GEN_POSTAGS"].str.contains(l), l] = 1
        analysis_df.to_csv(dbout, sep="\t", index=False)

    @staticmethod
    def build_brown_pos_dict():
        pos_dict = defaultdict(set)
        for word, tag in brown.tagged_words():
            pos_dict[word.lower()].add(tag)
        return pos_dict


    def generalize_brown(self, brown_tags):
        general = set()
        for _n in brown_tags:
            if "-" in _n:
                n = _n.split("-")[0]
            else:
                n = _n
            if n == "NIL" or n.startswith("*"):
                #general.add(None)
                continue
            elif n == "HV":
                general.add("Verb")
            elif n.startswith("PP"):
                general.add("Pronoun")
            elif n.startswith("RB") or n.startswith("QL"):
                general.add("Adverb")
            elif n.startswith("BE") or n.startswith("MD"):
                general.add("Verb")
            elif n.startswith("AB"):
                general.add("Determiner")
            elif n.startswith("DO"):
                general.add("Verb")
            elif n.startswith("PN"):
                general.add("Pronoun")
            elif n.startswith("AT"):
                general.add("Determiner")
            else:
                general_tag = str(self.brown_to_general.loc[self.brown_to_general["Brown_POS"]==n, "General_POS"].values[0])
                general.add (general_tag)
        return general


    def generalize_nltk(self, nltk_tags):
        general = set()
        for n in nltk_tags:
            if n == "a" or n == "s":
                general.add("Adjective")
            elif n == "v":
                general.add("Verb")
            elif n == "n":
                general.add("Noun")
            elif n == "r":
                general.add("Adverb")
            else:
                raise KeyError("{} dont exist".format(n))
        return general


    def analyze_words(self, list_words):
        data_words = []
        for word in tqdm(list_words):
            brown_tags = self.postags_dict.get(word.lower(), set())
            wordnet_tags =set()
            synsets = wn.synsets(word)
            synonyms = set()
            antonyms = set()
            variants = set()
            definitions = []

            for syn in synsets:
                definitions.append(syn.definition())
                wordnet_tags.add(syn.pos())
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
                    if lemma.antonyms():
                        antonyms.update(a.name() for a in lemma.antonyms())
                    variants.add(lemma.name())
                    variants.update(der.name() for der in lemma.derivationally_related_forms())
            if len(wordnet_tags) == 0:
                pos_tag_nltk = nltk.pos_tag([word])[0][-1]
                wordnet_tags.add(pos_tag_nltk)
                if len(nltk.pos_tag([word]))>1:
                    print("WTF")
                    exit()
                gennltk_tags = self.generalize_brown(wordnet_tags)
            else:
                gennltk_tags = self.generalize_nltk(wordnet_tags)
            genbrown_tags = self.generalize_brown(brown_tags)
            
            word_metadata = {"word": word, "definitions": "; ".join(definitions), "synonyms": "; ".join(list(synonyms)),
                            "antonyms": "; ".join(list(antonyms)), "variants": "; ".join(list(variants)),
                            "brown_POSTAGS": ",".join(list(brown_tags)), "nltk_POSTAGS": ",".join(list(wordnet_tags)),
                            "brown_GENPOSTAGS": ",".join(list(genbrown_tags)), "nltk_GENPOSTAGS": ",".join(list(gennltk_tags)),
                            "GEN_POSTAGS": ",".join(list(genbrown_tags.union(gennltk_tags)))}
            data_words.append(word_metadata)
        return data_words

class MakeFrequencies():

    def __init__(self, map_pos):
        self.map_pos = pd.read_csv(map_pos)

    def make_general_freq(self, freq_file, map_out):
        df_freq = pd.read_csv(freq_file, sep="\t")
        del df_freq["Unnamed: 0"]
        df_freq = df_freq.merge(self.map_pos, left_on="PoS", right_on="CLAWS_Tag")
        gen_freq = df_freq.groupby(["General_POS"], as_index=False)["Freq"].sum()
        gen_freq = gen_freq.sort_values(by=["Freq"], ascending=False)
        gen_freq["Perc"] = gen_freq["Freq"]/sum(gen_freq["Freq"])
        gen_freq.to_csv(map_out, sep="\t", index=False)

    def make_spwr_freq(self, freq_file):
        df_freq = pd.read_csv(freq_file, sep="\t")
        del df_freq["Unnamed: 0"]
        df_freq = df_freq.merge(self.map_pos, left_on="PoS", right_on="CLAWS_Tag")

        sp_freq = df_freq.groupby(["General_POS"], as_index=False)["FrSp"].sum()
        sp_freq = sp_freq.rename(columns={"FrSp": "Freq"})
        sp_freq = sp_freq.sort_values(by=["Freq"], ascending=False)
        sp_freq["Perc"] = sp_freq["Freq"]/sum(sp_freq["Freq"])
        sp_freq.to_csv("general_POS_freq_speak.txt", sep="\t", index=False)
    
        sp_freq = df_freq.groupby(["General_POS"], as_index=False)["FrWr"].sum()
        sp_freq = sp_freq.rename(columns={"FrWr": "Freq"})
        sp_freq = sp_freq.sort_values(by=["Freq"], ascending=False)
        sp_freq["Perc"] = sp_freq["Freq"]/sum(sp_freq["Freq"])
        sp_freq.to_csv("general_POS_freq_written.txt", sep="\t", index=False)

    def make_convtask_freq(self, freq_file):
        df_freq = pd.read_csv(freq_file, sep="\t")
        del df_freq["Unnamed: 0"]
        df_freq = df_freq.merge(self.map_pos, left_on="PoS", right_on="CLAWS_Tag")

        sp_freq = df_freq.groupby(["General_POS"], as_index=False)["FrDe"].sum()
        sp_freq = sp_freq.rename(columns={"FrDe": "Freq"})
        sp_freq = sp_freq.sort_values(by=["Freq"], ascending=False)
        sp_freq["Perc"] = sp_freq["Freq"]/sum(sp_freq["Freq"])
        sp_freq.to_csv("general_POS_freq_conversation.txt", sep="\t", index=False)
    
        sp_freq = df_freq.groupby(["General_POS"], as_index=False)["FrCg"].sum()
        sp_freq = sp_freq.rename(columns={"FrCg": "Freq"})
        sp_freq = sp_freq.sort_values(by=["Freq"], ascending=False)
        sp_freq["Perc"] = sp_freq["Freq"]/sum(sp_freq["Freq"])
        sp_freq.to_csv("general_POS_freq_task.txt", sep="\t", index=False)

    def make_imginf_freq(self, freq_file):
        df_freq = pd.read_csv(freq_file, sep="\t")
        del df_freq["Unnamed: 0"]
        df_freq = df_freq.merge(self.map_pos, left_on="PoS", right_on="CLAWS_Tag")

        sp_freq = df_freq.groupby(["General_POS"], as_index=False)["FrIm"].sum()
        sp_freq = sp_freq.rename(columns={"FrIm": "Freq"})
        sp_freq = sp_freq.sort_values(by=["Freq"], ascending=False)
        sp_freq["Perc"] = sp_freq["Freq"]/sum(sp_freq["Freq"])
        sp_freq.to_csv("general_POS_freq_imaginative.txt", sep="\t", index=False)
    
        sp_freq = df_freq.groupby(["General_POS"], as_index=False)["FrIn"].sum()
        sp_freq = sp_freq.rename(columns={"FrIn": "Freq"})
        sp_freq = sp_freq.sort_values(by=["Freq"], ascending=False)
        sp_freq["Perc"] = sp_freq["Freq"]/sum(sp_freq["Freq"])
        sp_freq.to_csv("general_POS_freq_informative.txt", sep="\t", index=False)


if __name__== "__main__":
    df = PolishDB(pos_tags="./pos_tags.tsv", freqs_tags="freqs_tags.tsv", brown_to_gen="./brown_to_general_pos.csv")
    df.makeDF(basefolder="../../../../data/gregg_definitive/", dbout="greggs_metadata.tsv")
    exit()
    m = MakeFrequencies(map_pos="claws_to_general_pos.csv")
    #m.make_general_freq(freq_file="postags_freq_all.txt", map_out="general_POS_freq_all.txt")
    #m.make_spwr_freq(freq_file="postags_freq_spwrit.txt")
    #m.make_convtask_freq(freq_file="postags_freq_taskconv.txt")
    m.make_imginf_freq(freq_file="postags_freq_imaginf.txt")

    