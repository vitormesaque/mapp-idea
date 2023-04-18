from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk
import faiss
import numpy as np
import bentoml
from bentoml.io import JSON
from math import ceil
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from transformers import AutoTokenizer
from scipy.special import softmax
from numpy.linalg import norm

ISSUES_PATH = "issues"
MODEL_PATH = "distilbert-multilingual-onnx"
ENCODING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

nltk.download('punkt')

class IssuesRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.issues = pd.read_pickle(f'{ISSUES_PATH}/issues.pkl')
        self.issues.fillna(value='NaN',inplace=True)
        self.encoder = SentenceTransformer(ENCODING_MODEL_NAME)
        self.lex_analizer = SentimentIntensityAnalyzer()
        self.onnx_model_path = f'{MODEL_PATH}/model.onnx'
        self.onnx_session = self._create_onnx_session() # don't init NN model if not needed
        self.tokenizer = AutoTokenizer.from_pretrained(f'{MODEL_PATH}')
        self.maxlen = 8

    def generate_ngrams(self, snippet, n_gram_size, n_gram_jump):
        counter = 0
        tmp_ngram_list = []
        splited_text = snippet.split()
        if len(splited_text) < n_gram_size:
            tmp_ngram_list.append(snippet)
        else:
            for grams in ngrams(splited_text, n_gram_size):
                if (counter % n_gram_jump) == 0:
                    tmp_ngram_list.append(' '.join(grams))
                counter += 1
            tmp_ngram_list.append(' '.join(grams))
            
        return list(set(tmp_ngram_list))

    def setentiment_pre_filtering(self, df_review):
        df_review['snippet_sentiment'] = df_review['snippet'].apply(lambda x: self.lex_analizer.polarity_scores(x)['compound'])
        return df_review[df_review['snippet_sentiment'] <= 0].reset_index(drop=True)

    def _create_onnx_session(self, provider="CPUExecutionProvider"):
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 0
        options.inter_op_num_threads = 0
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        session = InferenceSession(self.onnx_model_path, options, providers=[provider])
        session.disable_fallback()
        return session
    
    def NN_predict_sentiment(self, cur_str):
        tokens = self.tokenizer.encode_plus(cur_str, max_length=self.maxlen, truncation=True, 
                                            padding=False, return_tensors='tf', return_token_type_ids=False, 
                                            add_special_tokens=True)
        tokens = {name: np.atleast_2d(value).astype(np.int64) for name, value in tokens.items()}
        out = self.onnx_session.run(None,tokens)
        prob = softmax(out, axis=-1)[0][0]
        pred = np.argmax(prob)
        return pred - 1, prob[pred]

    def setentiment_pos_filtering(self, dict_match_ret):
        filter_dict = defaultdict(lambda : [])
        for cur_key in dict_match_ret:
            for i, cur_entity in enumerate(dict_match_ret[cur_key]):
                nn_sentiment_pred, nn_sentiment_prob  = self.NN_predict_sentiment(cur_entity['entity'])
                if nn_sentiment_pred == -1:
                    dict_match_ret[cur_key][i]['entity_sentiment'] = nn_sentiment_pred
                    dict_match_ret[cur_key][i]['entity_sentiment_confidence'] = nn_sentiment_prob
                    filter_dict[cur_key].append(dict_match_ret[cur_key][i])
        return dict(filter_dict)

    def encode_sentences(self, sentences):
        uniq_sentences = np.unique(sentences).tolist()
        chunk_size = 4
        map_sentence_to_emb = {}
        for i in range(0, len(uniq_sentences), chunk_size):
            cur_sentences = uniq_sentences[i:i + chunk_size]
            cur_embeddings = self.encoder.encode(cur_sentences, batch_size=chunk_size)  
            for cur_sentence, cur_emb in zip(cur_sentences, cur_embeddings):
                map_sentence_to_emb[cur_sentence] = cur_emb / norm(cur_emb)
        return [map_sentence_to_emb[cur_sentence] for cur_sentence in sentences]  
    
    def generate_search(self, df_review, min_association_thold, pre_lex_filter, pos_NN_filter):

        emb_index = np.array(self.issues['embeddings'].values.tolist())
        dimension = emb_index[0].shape[0]
        my_index = faiss.IndexFlatIP(dimension)
        my_index.add(emb_index)

        reviews_emb = np.array(df_review['embeddings'].values.tolist())
        reviews_ngram = df_review['snippet_ngram'].values
        reviews_review = df_review['text'].values
        distances, indexes = my_index.search(reviews_emb, k=1)

        if pre_lex_filter :
            snippet_sentiments = df_review['snippet_sentiment'].values
            counter = 0

        dict_match_ret = defaultdict(lambda : [])
        for cur_distance, cur_idx, cur_ngram, cur_review in zip(distances, indexes, reviews_ngram, reviews_review):
            cur_idx_0 = cur_idx[0]
            cur_row_issues = self.issues.iloc[cur_idx_0]
            
            if cur_distance[0] > min_association_thold:
                dict_atributes_ret = {
                    "entity" : cur_ngram,
                    "issue" : cur_row_issues['issue'],
                    "issue_dist" : cur_distance[0],
                    "issue_tier" : cur_row_issues['tier'],
                    "issue_tier_2" : cur_row_issues['tier_2_neigh'],
                    "issue_tier_2_dist" : cur_row_issues['tier_2_dist'],
                    "issue_tier_1" : cur_row_issues['tier_1_neigh'],
                    "issue_tier_1_dist" : cur_row_issues['tier_1_dist']
                }
                if pre_lex_filter :
                    dict_atributes_ret['snippet_sentiment'] = snippet_sentiments[counter]
                else:
                    dict_atributes_ret['snippet_sentiment'] = "NaN"
                if not pos_NN_filter:
                    dict_atributes_ret['entity_sentiment'] = "NaN"
                    dict_atributes_ret['entity_sentiment_confidence'] = "NaN"
                dict_match_ret[cur_review].append(dict_atributes_ret)
            if pre_lex_filter:
                counter += 1
        return dict(dict_match_ret)

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def predict(self, input, correlation_thold, n_gram_size, n_gram_jump, pre_lex_filter, pos_NN_filter):

        df_reviews = pd.DataFrame({'text': input})
        df_reviews['snippet'] = df_reviews['text'].apply(lambda x: sent_tokenize(x))
        df_reviews = df_reviews.explode('snippet').reset_index(drop=True)
        if pre_lex_filter:
            df_reviews = self.setentiment_pre_filtering(df_reviews)

        df_reviews['snippet_ngram'] = df_reviews['snippet'].apply(lambda x: self.generate_ngrams(x, n_gram_size, n_gram_jump))
        df_reviews = df_reviews.explode('snippet_ngram').reset_index(drop=True)
        
        df_reviews['embeddings'] = self.encode_sentences(df_reviews['snippet_ngram'].values)

        correlated_snippets = self.generate_search(df_reviews, correlation_thold, pre_lex_filter, pos_NN_filter)
        
        if pos_NN_filter:
            correlated_snippets = self.setentiment_pos_filtering(correlated_snippets)
        return correlated_snippets
    

model_runner = bentoml.Runner(IssuesRunnable)
svc = bentoml.Service("correlation", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
def predict(input):
    """"""
    batch_size = 4096

    pre_lex_filter = input['pre_lex_filter'] # True
    pos_NN_filter = input['pos_NN_filter'] # True
    correlation_thold = input['correlation_thold'] # 0.8
    n_gram_size = input['n_gram_size'] # 7
    n_gram_jump = input['n_gram_jump'] # 5


    input_texts = input['text']
    chunk_n = ceil(len(input_texts) / batch_size)
    inter = np.array_split(input_texts, chunk_n)
    response = []

    for chunk in inter:
        model_input = [x["text"] for x in chunk]
        predictions = model_runner.predict.run(model_input, correlation_thold, n_gram_size, n_gram_jump, pre_lex_filter, pos_NN_filter)
        for cur_text in predictions:
            response.append({
                "text" : cur_text,
                "entities" : predictions[cur_text]
            })
            
    return response
