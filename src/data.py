import os
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset

class RapDataset:
    def __init__(self, bpe_dir='../bpe', seq_len=512, emb_dim=512, max_token_ind=20000, batch_size=32, max_len=2048):
        self.r = r'\[(.*)\]\n'

        self.bpe_dir = bpe_dir
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.max_token_ind = 20000
        self.batch_size = 32
        self.max_len = max_len

        self.file_labels, self.file_array, self.tokens_per_file = self._load_files()
        self.samples_per_files = [int(np.ceil(x / self.seq_len)) for x in self.tokens_per_file]

    def _load_files(self):
        labels = os.listdir(self.bpe_dir)
        data_array = []
        valid_labels = []
        tokens_per_file = []
        print('Loading files')
        for file in tqdm(labels):
            with open(os.path.join(self.bpe_dir, file), 'r') as f:
                data = f.read()
            num_tokens = self._get_tokens_per_file(data)
            if num_tokens <= self.max_len:
                valid_labels.append(file)
                data_array.append(data)
                tokens_per_file.append(num_tokens)

        return valid_labels, data_array, tokens_per_file

    def _get_tokens_per_file(self, file):
        tokens = re.sub(self.r, '', file).strip().split(' ')
        return len(tokens)

    # def get_iterator(self):


    # class DataIterator:
    #     def __init__(self, n_samples, tokens_per_sample, batch_size, data, r):
    #         self.n_samples = n_samples
    #         self.tokens_per_sample = tokens_per_sample
    #         self.cnt = Counter(tokens_per_sample).most_commont()
    #         self.num_batches = {k: v // batch_size for k, v in self.cnt.items()}
    #
    #         self.data = data
    #
    #     def __iter__(self):
    #         return self
    #
    #     def __next__(self):
    #         choice_array = []
    #         for k, v in self.num_batches.items():
    #             choice_array.extend([k] * v)
    #         choice = np.random.choice(choice_array)
    #         self.num_batches[choice] -= 1
    #         if self.i < self.n:
    #             i = self.i
    #             self.i += 1
    #             return i
    #         else:
    #             raise StopIteration()

    # def __iter__(self):
