import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class E5_base_v2():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('E5model')
        self.model = AutoModel.from_pretrained('E5model')
        # self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
        # self.model = AutoModel.from_pretrained('intfloat/e5-base-v2')
        # self.tokenizer = torch.load('./E5model/tokenizer')
        # self.model = torch.load('./E5model/e5_model.model')


    def get_score(self, input_texts):
        n = len(input_texts)
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        # print(len(embeddings[0]))
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        # print(scores.tolist())
        # embeddings = F.normalize(embeddings, p=2, dim=1)
        # scores = (embeddings[:] @ embeddings[:].T) * 100
        # res = [scores.tolist()[i][i] for i in range(n)]
        # print(scores.tolist())
        return scores.tolist()

if __name__ == '__main__':
    e5_base = E5_base_v2()
    input_texts = ['query: how much protein should a female eat',
                   'query: summit define',
                   ]
    res = e5_base.get_score(input_texts)
    print(res)
