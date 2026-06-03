import csv
import difflib
import os
from pathlib import Path

import spacy


_POS_COLUMNS = ("Noun", "Verb", "Adjective", "Adverb", "Preposition", "Foreign",
                "Pronoun", "Interjection", "Conjunction", "Determiner", "Numeral",
                "Particle", "Other", "Existential")


class _WordIndex:
    """Tiny replacement for the pandas DataFrame that DictMatch used to hold.

    Stores the full word list plus per-general-POS sublists. Lookup is O(1)
    dict access instead of a DataFrame boolean-mask + tolist() round trip,
    and pandas is no longer a runtime dependency.
    """

    def __init__(self, all_words, pos_to_words):
        self.all_words = list(all_words)
        self.pos_to_words = pos_to_words

    def __len__(self):
        return len(self.all_words)

    def words_for_pos(self, general_pos):
        return self.pos_to_words.get(general_pos, [])


def _parse_metadata_tsv(path):
    all_words = []
    pos_to_words = {col: [] for col in _POS_COLUMNS}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            word = row.get("word") or "None"
            all_words.append(word)
            for col in _POS_COLUMNS:
                if row.get(col) == "1":
                    pos_to_words[col].append(word)
    return _WordIndex(all_words, pos_to_words)


class DictMatch:

    spacy_to_general = {
        "ADJ": "Adjective", "ADP": "Preposition", "ADV": "Adverb", "AUX": "Verb",
        "CONJ": "Conjunction", "CCONJ": "Conjunction", "DET":"Determiner", "NOUN": "Noun",
        "NUM": "Numeral", "PART": "Particle", "PRON": "Pronoun", "PROPN": "Noun",
        "SCONJ": "Conjunction", "VERB": "Verb", "X": "Other", "MOD":"Verb", "PREP": "Preposition",
        "REL": "Pronoun"}

    def __init__(self, db_words):
        self.nlp = spacy.load("en_core_web_sm")
        self.valid_words = DictMatch.load_valid_words(db_words=db_words)

    @staticmethod
    def load_valid_words(db_words):
        if isinstance(db_words, str) and os.path.isfile(db_words):
            return _parse_metadata_tsv(db_words)
        if isinstance(db_words, str) and os.path.isdir(db_words):
            words = [f.replace(".png", "") for f in os.listdir(db_words)]
            return _WordIndex(words, {col: [] for col in _POS_COLUMNS})
        if not db_words:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            return _parse_metadata_tsv(
                Path(current_directory) / "../data/greggs_metadata.tsv")
        if isinstance(db_words, list):
            return _WordIndex(db_words, {col: [] for col in _POS_COLUMNS})
        raise ValueError("DB_words is not valid")

    def vanilla_distance_annotations(self, candidates, n=1):
        if len(self.valid_words) == 0:
            raise KeyError("No valid words available")
        closest_matches = {}
        for word in candidates:
            match = difflib.get_close_matches(word, self.valid_words.all_words,
                                              n=n, cutoff=0.0)
            closest_matches[word] = match if match else None
        return closest_matches

    def predict_pos(self, word):
        """Predicts the POS tag of a given word using spaCy."""
        doc = self.nlp(word)
        return doc[0].pos_ if doc else None

    def find_closest_POSmatch(self, predicted_word, predicted_pos=None, n=3, cutoff=0.0):
        """Closest dictionary word to predicted_word, restricted to the same POS."""
        if predicted_pos is None:
            dictcandidates = self.valid_words.all_words
        else:
            try:
                general_pos = DictMatch.spacy_to_general[predicted_pos]
            except KeyError:
                # Original behaviour: spaCy POS tag not in the map (INTJ,
                # PUNCT, SYM, etc.) falls back to all words.
                dictcandidates = self.valid_words.all_words
            else:
                # If the POS bucket is empty, leave it empty so difflib
                # returns no match -- matches the original DataFrame
                # boolean-mask behaviour. Do NOT fall back to all_words here.
                dictcandidates = self.valid_words.words_for_pos(general_pos)
        return difflib.get_close_matches(predicted_word, dictcandidates, n=n, cutoff=cutoff)

# Example usage
if __name__ == "__main__":
    # Sample dictionary
    dictionary = ["run", "apple", "beautiful", "quickly", "dog", "eat", "happy", "tree", "jump", "slowly"]
    from luvia.straw.straw import Straw

    # Tag the dictionary
    straw = Straw()
    tagged_dict = straw.valid_words
    d_pos = DictMatch(tagged_dict)

    # Example predicted word
    predicted_word = "runn"
    predicted_pos = d_pos.predict_pos(predicted_word)

    # Find closest match
    match = d_pos.find_closest_match(predicted_word, predicted_pos)

    print(f"Predicted word: {predicted_word}")
    print(f"Predicted POS: {predicted_pos}")
    print(f"Closest match with same POS: {match}")
