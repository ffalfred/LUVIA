import pandas as pd
import spacy
import difflib
import os
from pathlib import Path

class DictMatch:

    spacy_to_general = {
        "ADJ": "Adjective", "ADP": "Preposition", "ADV": "Adverb", "AUX": "Verb",
        "CONJ": "Conjunction", "CCONJ": "Conjunction", "DET":"Determiner", "NOUN": "Noun",
        "NUM": "Numeral", "PART": "Particle", "PRON": "Pronoun", "PROPN": "Noun",
        "SCONJ": "Conjunction", "VERB": "Verb", "X": "Other", "MOD":"Verb", "PREP": "Preposition",
        "REL": "Pronoun"}

    def __init__(self, db_words):

        self.nlp =  spacy.load("en_core_web_sm")
        self.valid_words = DictMatch.load_valid_words(db_words=db_words)

    @staticmethod
    def load_valid_words(db_words):
        valid_words = None
        if os.path.isfile(db_words):
            df_words = pd.read_csv(db_words, sep="\t", dtype={"word": str})
            df_words.loc[df_words["word"].isna(), "word"] = "None"
            valid_words.extend(df_words["word"].tolist())
        elif os.path.isdir(db_words):
            words_lst = []
            for filename in os.listdir(db_words):
                words_lst.append(filename.replace(".png", ""))
            valid_words = pd.DataFrame(words_lst)
        elif not db_words:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            filedf = Path(current_directory) / '../data/greggs_metadata.tsv'
            df_words = pd.read_csv(filedf, sep="\t", dtype={"word": str})
            df_words.loc[df_words["word"].isna(), "word"] = "None"
            valid_words = df_words
        elif isinstance(db_words, list):
            valid_words = pd.DataFrame(db_words)
        else:
            raise ValueError("DB_words is not valid")
        return valid_words

    def vanilla_distance_annotations(self, candidates, n=1):
        if len(self.valid_words) == 0:
            raise KeyError("No valid words availables")
        closest_matches = {}
        for word in candidates:
            # Use difflib to find the closest match
            match = difflib.get_close_matches(word, self.valid_words["word"].tolist(),
                                        n=n, cutoff=0.0)
            if match:
                closest_matches[word] = match
            else:
                closest_matches[word] = None
        return closest_matches

    def predict_pos(self, word):
        """
        Predicts the POS tag of a given word using spaCy.
        """
        doc = self.nlp(word)
        return doc[0].pos_ if doc else None

    def find_closest_POSmatch(self, predicted_word, predicted_pos=None, n=3,
                                cutoff=0.0):
        """
        Finds the closest dictionary word to the predicted word with the same POS tag.
        Uses string similarity (difflib) for matching.
        """
        # Filter dictionary by POS
        #candidates = [word for word, pos in tagged_dict.items() if pos == predicted_pos]
        if predicted_pos is not None:
            dictcandidates = self.valid_words.loc[self.valid_words[DictMatch.spacy_to_general[predicted_pos]]==1, "word"].tolist()
        else:
            print("NOOO")
            exit()
            dictcandidates = self.valid_words["word"].tolist()
        # Use difflib to find closest match
        matches = difflib.get_close_matches(predicted_word, dictcandidates, n=n, cutoff=cutoff)
        return matches

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
