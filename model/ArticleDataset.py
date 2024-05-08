from torch.utils.data import DataLoader, Dataset

# 아티클 데이터셋
class ArticleDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.texts = dataframe['contents'].tolist()
        self.targets = dataframe['label'].tolist()
        self.lengths = dataframe['origin_len'].tolist()
        self.writedAt = dataframe[['writedAt_n_2019Q1', 'writedAt_n_2019Q2', 'writedAt_n_2019Q3', 'writedAt_n_2019Q4',
                                   'writedAt_n_2020Q1', 'writedAt_n_2024Q1', 'writedAt_n_2024Q2']].values
        self.scrapedAt = dataframe[['scrapedAt_n_2024Q2']].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        length = self.lengths[idx]
        writedAt = self.writedAt[idx]
        scrapedAt = self.scrapedAt[idx]
        return text, length, writedAt, scrapedAt, target
