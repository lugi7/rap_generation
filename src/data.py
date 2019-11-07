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
        self.samples_per_file = [int(np.ceil(x / self.seq_len)) for x in self.tokens_per_file]

        self.geniusid2local = np.load('../geniusid2local.npy').item()
        self.part2ind = np.load('../part2ind.npy').item()

    def _load_files(self):
        labels = os.listdir(self.bpe_dir)
        data_array = []
        valid_labels = []
        tokens_per_file = []
        print('Loading files')
        for file in tqdm(labels[:1000]):
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

    def __iter__(self):
        self.cnt = Counter(self.samples_per_file)
        self.num_batches = {k: v // self.batch_size for k, v in self.cnt.items()}
        self.sample_inds = {k: list(np.where(np.array(self.samples_per_file) == k)[0]) for k in self.cnt.keys()}
        return self

    def __next__(self):
        choice_array = []
        for k, v in self.num_batches.items():
            choice_array.extend([k] * v)
        if not choice_array:
            raise StopIteration()

        choice = np.random.choice(choice_array)
        self.num_batches[choice] -= 1

        inds = self.sample_inds[choice]
        inds_to_take = list(np.random.choice(inds, self.batch_size))
        for ind in inds_to_take:
            print(ind)
            print(inds)
            inds.remove(ind)
        batch_files = [self.file_array[x] for x in inds_to_take]
        batch = self._files2numeric(batch_files, choice)

        return batch

    def _files2numeric(self, batch, choice):
        #returns choice * ((bs, seq_len), (bs, seq_len, n_parts), (bs, seq_len, num_artists))
        tokens, parts, artists = zip(*[self._prepare_sample(b, choice) for b in batch])

        tokens = np.split(np.stack(tokens), choice, 1)
        parts = np.split(np.stack(parts), choice, 1)
        artists = np.split(np.stack(artists), choice, 1)

        return list(zip(tokens, parts, artists))

    def _prepare_sample(self, sample, choice):
        tokens = np.full((choice * self.seq_len), fill_value=self.max_token_ind, dtype=np.long)
        parts = np.zeros((choice * self.seq_len, max(self.part2ind.values()) + 1), dtype=np.float32)
        artists = np.zeros((choice * self.seq_len, len(self.geniusid2local)), dtype=np.float32)

        tokens_split = re.split(self.r, sample)[1:]
        assert len(tokens_split) % 2 == 0

        segments = len(tokens_split) // 2

        last_pos = 0

        for seg in range(segments):
            cc = tokens_split[2 * seg].split(',')
            cc = [x.strip() for x in cc]

            tokens_string = tokens_split[2 * seg + 1]
            tokens_string = tokens_string.strip().split()
            tokens_int = [int(x) for x in tokens_string]
            curr_len = len(tokens_int)

            tokens[last_pos: last_pos + curr_len] = tokens_int

            for c in cc:
                try:
                    part_ind = self.part2ind[c]
                    parts[last_pos: last_pos + curr_len, part_ind] = 1.
                except:
                    pass

                try:
                    artist_ind = self.geniusid2local[int(c)]
                    artists[last_pos: last_pos + curr_len, artist_ind] = 1.
                except:
                    pass

            last_pos += curr_len

        return tokens, parts, artists