import random
import math
from itertools import product
import spacy
from nltk.corpus import wordnet as wn
import nltk
#nltk.download('wordnet')
from tqdm import tqdm
import os
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class Tongue:

    def __init__(self):

        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    
    def create_sentences(self, word_buckets, sample_min=20):
        # Generate a sample of sentences
        all_combinations = list(product(*word_buckets))
        sampled_sentences = random.sample(all_combinations, min(sample_min, len(all_combinations)))
        sentences = [' '.join(words) for words in sampled_sentences]
        return sentences

    # Function to calculate perplexity
    def calculate_perplexity(self,sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)

    def analyze_sentences(self, sentences):
        # Analyze sentences
        results = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            syntax_info = [(token.text, token.dep_, token.head.text) for token in doc]
            perplexity = self.calculate_perplexity(sentence)
            results.append({
                "sentence": sentence,
                "syntax": syntax_info,
                "perplexity": perplexity})
        return results

    @staticmethod
    def analyze_words(list_words):
        analysis = {}
        for word in list_words:
            synsets = wn.synsets(word)
            analysis[word]={"pos_tags":set(), "verb_frames":set(), "noun_types":set(), "variants":set()}
            for syn in synsets:
                pos_tags.add(syn.pos())
                # Verb frames
                if pos == 'v':
                    for frame in syn.frame_strings():
                        verb_frames.add(frame)
                # Noun types (hypernyms)
                if pos == 'n':
                    for hypernym in syn.hypernyms():
                        noun_types.add(hypernym.name().split('.')[0])
                # Morphological variants
                lemma_names = syn.lemma_names()
                for lemma in lemma_names:
                    variants.add(lemmatizer.lemmatize(lemma, pos=pos))
            analysis[word]["pos_tags"].add(pos_tags)
            analysis[word]["verb_frames"].add(verb_frames)
            analysis[word]["noun_types"].add(noun_types)
            analysis[word]["variants"].add(variants)

        return analysis

if __name__== "__main__":
    list_words = []
 #   for filename in os.listdir("../../../data/gregg_definitive/"):
  #      list_words.append(filename.replace(".png", ""))
  #  analysis = Tongue.analyze_words(list_words)
  #  forms = {"a":0, "s":0, "v":0, "n":0, "r":0, "Non":0}
  #  for k, val in tqdm(analysis.items()):
  #      if len(val["pos_tags"]) == 0:
  #          forms["Non"] += 1
  #      else:
   #         for key in forms:
    #            if key in val:
     #               forms[key] += 1
    #print(forms) 
    word_buckets = [['content', 'connive','medication', 'dedication'],['fork', 'for', 'france', 'romance', 'france', 'performance'],
                    ['mother', 'smother', 'meter', 'meteor']]
    tongue = Tongue()
    sentences = tongue.create_sentences(word_buckets=word_buckets)
    analyze_sentences = tongue.analyze_sentences(sentences=sentences)
    print(analyze_sentences)    
    # Sort by perplexity
    analyze_sentences.sort(key=lambda x: x["perplexity"])
    print(analyze_sentences)

