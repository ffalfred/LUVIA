import random
import math
from itertools import product
import spacy
from nltk.corpus import wordnet as wn
import nltk
from pathlib import Path
from tqdm import tqdm
import os
import torch
import json
import random
import numpy as np

#from gramformer import Gramformer
from luvia.tongue.distance import DictMatch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from deepmultilingualpunctuation import PunctuationModel


class Tongue:

    def __init__(self, db_words=False, match_mode=False, character="random"):

        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")

        # Load GPT-2 model and tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        if match_mode:
            self.dictmatch_module = DictMatch(db_words=db_words)
            if match_mode == "character_POS":
                self.character_features = Tongue.load_character(character)
            else:
                self.character_features = None
        else:
            self.dictmatch_module = None
        self.match_mode = match_mode
        self.max_length_structure = 7

    @staticmethod
    def load_valid_words(db_words):
        valid_words = []
        if os.path.isfile(db_words):
            df_words = pd.read_csv(db_words, sep="\t", dtype={"word": str})
            df_words.loc[df_words["word"].isna(), "word"] = "None"
            valid_words.extend(df_words["word"].tolist())
        elif os.path.isdir(db_words):
            for filename in os.listdir(db_words):
                valid_words.append(filename.replace(".png", ""))
        elif not db_words:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            filedf = Path(current_directory) / '../data/greggs_metadata.tsv'
            df_words = pd.read_csv(filedf, sep="\t", dtype={"word": str})
            df_words.loc[df_words["word"].isna(), "word"] = "None"
            valid_words.extend(df_words["word"].tolist())
        elif isinstance(db_words, list):
            valid_words.extend(db_words)
        else:
            raise ValueError("DB_words is not valid")
        return valid_words

    @staticmethod
    def load_character(name):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        folder_characs = Path(current_directory) / '../data/characters/'

        if name == "random":
            files = os.listdir(folder_characs)
            file_chosen = random.choice(files)
        else:
            file_chosen = "{}_syntax_profile.json".format(name)
        with open("{}/{}".format(folder_characs, file_chosen), "r") as fj:
            character_features = json.load(fj)
        character_features["all_templates"] = []
        for k, val in character_features["templates"].items():
            character_features["all_templates"].extend(val)
        return character_features

    def create_sentences(self, word_buckets, sample_min=40):
        final_buckets = []
        for wbucket in word_buckets:
            bucket = []
            for k, val in wbucket.items():
                bucket.extend(val)
            final_buckets.append(bucket)
        # Generate a sample of sentences
        all_combinations = list(product(*final_buckets))
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


    @staticmethod
    def syntactic_score(sentence):
        doc = self.nlp(sentence)
        pos_tags = [token.pos_ for token in doc]
        words = [token.text.lower() for token in doc]

        score = 0

        # Penalize consecutive nouns
        score -= sum(1.5 for i in range(1, len(pos_tags)) if pos_tags[i] == "NOUN" and pos_tags[i - 1] == "NOUN")
        # Reward POS diversity
        score += len(set(pos_tags)) * 1.0

        # Ensure presence of VERB and DET
        score += 2.0 if "VERB" in pos_tags else -2.0
        score += 1.0 if "DET" in pos_tags else -1.0

        # Penalize repeated words
        word_freq = {word: words.count(word) for word in set(words)}
        score -= sum(1.0 for count in word_freq.values() if count > 1)

        # Reward balanced use of content and function words
        content_pos = {"NOUN", "VERB", "ADJ", "ADV"}
        function_pos = {"DET", "PRON", "ADP", "AUX", "CONJ", "CCONJ", "PART", "SCONJ"}
        content_count = sum(1 for pos in pos_tags if pos in content_pos)
        function_count = sum(1 for pos in pos_tags if pos in function_pos)
        if content_count > 0 and function_count > 0:
            balance_ratio = min(content_count, function_count) / max(content_count, function_count)
            score += balance_ratio * 2.0

        return score

    @staticmethod
    def rerank_beam_search(candidates, syntax_weight=1.0):
        scored_candidates = []
        for sentence, log_likelihood in candidates:
            syntax_score = Tongue.syntactic_score(sentence)
            combined_score = log_likelihood + syntax_weight * syntax_score
            scored_candidates.append((sentence, combined_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[0][0]

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
        lemmatizer = self.ntlk.WordNetLemmatizer()
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

    def dictmatch_vanilla(self, sentence, candidates):
        refined_sentence = []
        for idx, words in enumerate(sentence):
            matchwords = self.dictmatch_module.vanilla_distance_annotations(words,
                                                                        candidates)
            refined_sentence.append(matchwords)
        return refined_sentence

    def dictmatch_samePOS(self, sentence, candidates):
        refined_sentence = []
        for idx, bucket in enumerate(sentence):
            refined_bucket = {}
            for word in bucket:
                wordpos = self.dictmatch_module.predict_pos(word)
                refined_words = self.dictmatch_module.find_closest_POSmatch(
                                            word, wordpos, n=candidates)
                refined_bucket[word] = refined_words
            refined_sentence.append(refined_bucket)
        return refined_sentence

    def dictmatch_schemePOS(self, sentence, candidates):
        length_sentence = len(sentence)
        refined_words = []
        possible_structures = []
        for structure in self.character_features["all_templates"]:
            if len(structure) == length_sentence:
                possible_structures.append(structure)
            elif length_sentence > self.max_length_structure and len(structure) == self.max_length_structure:
                _structure = []
                i = 0
                j = 0
                while True:
                    _structure.append(structure[i])
                    i += 1
                    if i >= len(structure):
                        i = 0
                    if len(_structure) >= length_sentence:
                        break
                possible_structures.append(_structure)
            else:
                continue
        structure_sel = random.choice(possible_structures)
        refined_sentence = []
        for idx, bucket in enumerate(sentence):
            refined_bucket = {}
            for word in bucket:
                refined_words = self.dictmatch_module.find_closest_POSmatch(
                                                word, structure_sel[idx], n=candidates)
                refined_bucket[word] = refined_words
            refined_sentence.append(refined_bucket)
        return refined_sentence

    def finetune_inference(self, sentence, candidates=3):
        if not self.match_mode:
            refined_sentence = sentence
        elif self.match_mode == "vanilla":
            refined_sentence = self.dictmatch_vanilla(sentence, candidates)
        elif self.match_mode == "equal_POS":
            refined_sentence = self.dictmatch_samePOS(sentence, candidates)
        elif self.match_mode == "character_POS":
            refined_sentence = self.dictmatch_schemePOS(sentence, candidates)
        return refined_sentence
    
    def punctuate(self, sentences, num_variants=5, temperature=1.):

        model = PunctuationModel()
        sentence_variants = []
        for sentence in sentences:
            clean_text = model.preprocess(sentence)
            labeled_words = model.predict(clean_text)
            variants = []
            for _ in range(num_variants):
                variant = []
                for word, punct, confidence in labeled_words:
                    variant.append(word)
                    # Sample punctuation based on confidence
                    if punct != '0' and random.random() < confidence:
                        variant.append(punct)
                # End sentence with a stylistic punctuation
                variant.append(random.choice(['.', '!', '?']))
                variants.append(" ".join(variant))
            sentence_variants.append(variants)
        return sentence_variants


    def correct(self, sentences, correct_k):
        # Load model and tokenizer
        model_name = "prithivida/grammar_error_correcter_v1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        corrected_batch = []
        for sentence in sentences:
            input_text = f"gec: {sentence}"
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            outputs = model.generate(inputs, max_length=128, num_beams=correct_k,
                                        num_return_sequences=correct_k, early_stopping=True)
            corrected_sentences = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            corrected_batch.extend(corrected_sentences)
            corrected_batch.append(sentence)
        corrected_batch = list(set(corrected_batch))
        return corrected_batch

    def get_sentence(self, sentences, mode="best", k=1, quantile="5th"):
        if mode == "random":
            sentences_meta = random.sample(sentences, k)
        elif mode == "best":
            sentences.sort(key=lambda x: x["perplexity"])
            sentences_meta = sentences[:k]
        elif mode == "quantile":
            perplexity_values = []
            for entry in sentences:
                perplexity_values.append(entry["perplexity"])
            # Compute quantiles
            quantiles = {
                "5th": (np.percentile(perplexity_values, 0), np.percentile(perplexity_values, 5)),
                "10th": (np.percentile(perplexity_values, 5), np.percentile(perplexity_values, 10)),
                "25th": (np.percentile(perplexity_values, 10), np.percentile(perplexity_values, 25)),
                "50th": (np.percentile(perplexity_values, 25), np.percentile(perplexity_values, 50)),
                "75th": (np.percentile(perplexity_values, 50), np.percentile(perplexity_values, 75)),
                "90th": (np.percentile(perplexity_values, 75), np.percentile(perplexity_values, 90)),
                "95th": (np.percentile(perplexity_values, 90), np.percentile(perplexity_values, 95)),
                "100th": (np.percentile(perplexity_values, 95), np.percentile(perplexity_values, 100))}
            quantile_sel = quantiles[quantile]
            quantile_sentences = []
            for entry in sentences:
                if entry["perplexity"] >= quantile_sel[0] and entry["perplexity"] <= quantile_sel[1]:
                    quantile_sentences.append(entry)
            sentences_meta = random.sample(sentences, k)
        return sentences_meta

    

if __name__== "__main__":
    word_buckets = [['content', 'connive','medication', 'dedication'],['fork', 'for', 'france', 'romance', 'france', 'performance'],
                    ['mother', 'smother', 'meter', 'meteor']]
    tongue = Tongue(match_mode="character_POS")
    refined_word_buckets = tongue.finetune_inference(word_buckets)
    print(refined_word_buckets)
    proposed_sentences = tongue.create_sentences(refined_word_buckets)
    print(proposed_sentences)
    corrected_sentences = tongue.correct(proposed_sentences, correct_k=5)
    print(corrected_sentences)
    analyzed_sentences = tongue.analyze_sentences(corrected_sentences)
    print(analyzed_sentences)
    quantiled_sentences = tongue.get_sentence(analyzed_sentences,mode="quantile", k=2)
    print(quantiled_sentences)
    exit()
    sentences = tongue.create_sentences(word_buckets=word_buckets)
    analyze_sentences = tongue.analyze_sentences(sentences=sentences)
    print(analyze_sentences)    
    # Sort by perplexity
    analyze_sentences.sort(key=lambda x: x["perplexity"])
    print(analyze_sentences)

