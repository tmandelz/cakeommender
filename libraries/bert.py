import pandas as pd
import numpy as np
from enum import Enum
import torch
import transformers as ppb
import sentence_transformers as stf

# Classes
class BertEnum(Enum):
    """
    Thomas Mandelz
    Enum Class to BERT Models
    """

    BERT = 1
    SBERT = 2


class Bertomat(object):
    def __init__(self, bertmodel: BertEnum, pooling: str='cls'):
        self.bertmodel = bertmodel
        self.pooling = pooling

    def generateEmbeddings(self,data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BERT sentence embeddings / Joseph Weibel

        Uses texts in attribute 'overview' to generate sentence embeddings using the BERT model 'bert-base-uncased'. For that, the text is tokenized using a BERT tokenizer.

        A CUDA-GPU is used if available.

        :param pd.DataFrame data: meta data containing attribute 'overview' which is used to generate sentence embeddings and attribute 'movieId' used as index for the final DataFrame.
        :return: sentence embeddings as DataFrame with shape = (n_samples, 768) and movieId set as index
        :rtype: pd.DataFrame
        """

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if self.bertmodel == BertEnum.BERT:

            # tokenize texts
            tokenizer = ppb.BertTokenizer.from_pretrained("bert-base-uncased")
            tokenized = data.overview.apply(
                lambda x: tokenizer.encode(x, add_special_tokens=True)
            )

            # ensure all token vectors have same length and create matrix
            max_tokens = tokenized.apply(lambda x: len(x)).max()
            tokenized = tokenized.apply(
                lambda x: np.pad(x, (0, max_tokens - len(x)), mode="constant")
            )
            tokens = np.row_stack(tokenized.values)

            model = ppb.BertModel.from_pretrained("bert-base-uncased").to(device)

            with torch.no_grad():  # improve performance, no gradient needed
                # use model to build embeddings
                tokens = torch.from_numpy(tokens).to(device)
                embeddings = model(tokens)[0]

            if self.pooling == 'cls':
                # use embeddings of [CLS] token
                sentence_embeddings = embeddings[:,0,:]
            elif self.pooling == 'avg':
                # average all word embeddings
                sentence_embeddings = embeddings[:,1:,:].mean(axis=1)

            # detach tensor to move it back to CPU if it was on GPU
            return pd.DataFrame(sentence_embeddings.detach(), index=data.movieId)

        elif self.bertmodel == BertEnum.SBERT:
            # Get Sbert Model from sentence transformers
            sBertModel = stf.SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            # encode text to embeddings
            sentence_embeddings = sBertModel.encode(data.overview)
            return pd.DataFrame(sentence_embeddings, index=data.movieId)

