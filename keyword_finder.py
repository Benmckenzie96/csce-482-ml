import pandas as pd

class KeywordFinder:

    def __init__(self, org_dataset, org_vectorspace):
        self.dataset = org_dataset
        self.vs = org_vectorspace
        vocab = self.vs.get_vocabulary().keys()
        vocab_ind = self.vs.get_vocabulary().values()
        self.vocab_df = pd.DataFrame(vocab_ind, columns=['vocab_ind'])
        self.vocab_df['vocab'] = vocab
        self.vocab_df = self.vocab_df.sort_values(['vocab_ind'], ignore_index=True)

    def top_n_keywords(self, vector, n=5):
        df = self.vocab_df
        try:
            df['vec'] = vector
        except ValueError:
            df['vec'] = vector.reshape(self.vocab_df.shape[0], 1)
        df = df.sort_values(['vec'], ascending=False, ignore_index=True)
        return df.iloc[0:n]['vocab'].to_numpy().tolist()
