#!/usr/bin/env python3

from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import MiniBatchKMeans
from networkx.algorithms import community
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from collections import defaultdict
from scipy.special import softmax
from numpy.linalg import norm
from nltk.util import ngrams
from math import ceil
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import os
import nltk
import faiss

tqdm.pandas()

nltk.download('punkt', download_dir='/usr/local/share/nltk_data')
ISSUES_PATH = "model/issues" # don't change this !!!!
MODEL_PATH = "distilbert-multilingual-onnx"
ENCODING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

class IssueDetector():

    def __init__(self):

        self.issues = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'model', 'issues', 'issues.pkl'))

        self.issues.fillna(value='NaN',inplace=True)

        if 'severity' not in self.issues.columns:

            kmeans = MiniBatchKMeans(n_clusters=int(np.sqrt(len(self.issues))),
                                     random_state=0,
                                     batch_size=6,
                                     n_init=1)

            kmeans = kmeans.fit(np.array(self.issues.embeddings.to_list()))

            df_km_labels = pd.DataFrame(kmeans.labels_)

            for index,row in tqdm(self.issues.iterrows(),total=len(self.issues)):
                km_label = kmeans.labels_[index]
                cl_size = df_km_labels[df_km_labels[0]==km_label]
                self.issues.at[index,'severity'] = len(cl_size)


        self.issues.to_pickle(os.path.join(os.path.dirname(__file__), '..', 'model', 'issues', 'issues.pkl'))


        #self.encoder = SentenceTransformer(ENCODING_MODEL_NAME, device='cuda')
        self.encoder = SentenceTransformer(ENCODING_MODEL_NAME, device='cpu')
        self.lex_analizer = SentimentIntensityAnalyzer()
        self.onnx_model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'distilbert-multilingual-onnx', 'model.onnx')
        self.onnx_session = self._create_onnx_session() # don't init NN model if not needed
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(__file__), '..', 'model', 'distilbert-multilingual-onnx'))
        self.maxlen = 8
        self.issue_network = nx.read_gpickle(os.path.join(os.path.dirname(__file__), '..', 'model', 'issue_network.model'))


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
    #def _create_onnx_session(self, provider="CUDAExecutionProvider"):
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 0
        options.inter_op_num_threads = 0
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend
        session = InferenceSession(self.onnx_model_path, options, providers=[provider])
        session.disable_fallback()
        return session

    def NN_predict_sentiment(self, list_str):

        tokens = self.tokenizer.batch_encode_plus(list_str, max_length=self.maxlen, truncation=True,
                                            padding=True, return_tensors='tf', return_token_type_ids=False,
                                            add_special_tokens=True)
        tokens = {name: np.atleast_2d(value).astype(np.int64) for name, value in tokens.items()}
        out = self.onnx_session.run(None,tokens)
        prob = softmax(out, axis=-1)[0]
        pred = np.argmax(prob, axis=-1)

        return pred - 1, prob[0][pred]

    def setentiment_pos_filtering(self, dict_match_ret):
        filter_dict = defaultdict(lambda : [])
        for cur_key in tqdm(dict_match_ret, total=len(dict_match_ret)):
            nn_sentiment_pred, nn_sentiment_prob  = self.NN_predict_sentiment([cur_ent['entity'] for cur_ent  in dict_match_ret[cur_key]])
            i = 0
            for cur_nn_sentiment_pred, cur_nn_sentiment_prob in zip(nn_sentiment_pred, nn_sentiment_prob):
                if cur_nn_sentiment_pred == -1:
                    dict_match_ret[cur_key][i]['entity_sentiment'] = cur_nn_sentiment_pred
                    dict_match_ret[cur_key][i]['entity_sentiment_confidence'] = cur_nn_sentiment_prob
                    filter_dict[cur_key].append(dict_match_ret[cur_key][i])
                i+=1

        return dict(filter_dict)

    def encode_sentences(self, sentences):
        uniq_sentences = np.unique(sentences).tolist()
        chunk_size = 64
        map_sentence_to_emb = {}
        for i in tqdm(range(0, len(uniq_sentences), chunk_size), total= (len(uniq_sentences)//chunk_size + 1)):
            cur_sentences = uniq_sentences[i:i + chunk_size]
            cur_embeddings = self.encoder.encode(cur_sentences, batch_size=chunk_size)
            for cur_sentence, cur_emb in zip(cur_sentences, cur_embeddings):
                map_sentence_to_emb[cur_sentence] = cur_emb / norm(cur_emb)
        return [map_sentence_to_emb[cur_sentence] for cur_sentence in sentences]

    def generate_search(self, df_review, min_association_thold, pre_lex_filter, pos_NN_filter):

        print('-- generate search 1 -- ')
        emb_index = np.array(self.issues['embeddings'].values.tolist())
        dimension = emb_index[0].shape[0]

        my_index = faiss.IndexFlatIP(dimension)
        my_index.add(emb_index)

        print('-- generate search 2 -- ')

        reviews_emb = np.array(df_review['embeddings'].values.tolist())
        reviews_ngram = df_review['snippet_ngram'].values
        reviews_review = df_review['text'].values
        r_index = df_review['r_index'].values

        distances, indexes = my_index.search(reviews_emb, k=1)

        if pre_lex_filter :
            snippet_sentiments = df_review['snippet_sentiment'].values
            counter = 0

        dict_match_ret = defaultdict(lambda : [])
        for cur_distance, cur_idx, cur_ngram, cur_review, cur_r_index in zip(distances, indexes, reviews_ngram, reviews_review, r_index):
            cur_idx_0 = cur_idx[0]
            cur_row_issues = self.issues.iloc[cur_idx_0]

            if cur_distance[0] > min_association_thold:
                dict_atributes_ret = {
                    #"text" : cur_review,
                    "entity" : cur_ngram.strip(),
                    "issue" : cur_row_issues['issue'].strip(),
                    "issue_dist" : cur_distance[0],
                    "issue_tier" : cur_row_issues['tier'],
                    "issue_tier_2" : cur_row_issues['tier_2_neigh'].strip(),
                    "issue_tier_2_dist" : cur_row_issues['tier_2_dist'],
                    "issue_tier_1" : cur_row_issues['tier_1_neigh'].strip(),
                    "issue_tier_1_dist" : cur_row_issues['tier_1_dist'],
                    "severity" : cur_row_issues['severity']
                }
                if pre_lex_filter :
                    dict_atributes_ret['snippet_sentiment'] = snippet_sentiments[counter]
                else:
                    dict_atributes_ret['snippet_sentiment'] = "NaN"
                if not pos_NN_filter:
                    dict_atributes_ret['entity_sentiment'] = "NaN"
                    dict_atributes_ret['entity_sentiment_confidence'] = "NaN"
                dict_match_ret[cur_r_index].append(dict_atributes_ret)
            if pre_lex_filter:
                counter += 1
        return dict(dict_match_ret)


    def predict(self, input, correlation_thold, n_gram_size, n_gram_jump, pre_lex_filter, pos_NN_filter):
        print("--- predict init ---")
        #df_reviews = pd.DataFrame({'text': input})
        df_reviews = input
        print(df_reviews)
        df_reviews.drop_duplicates(subset='text',inplace=True)
        df_reviews.reset_index(inplace=True)
        print("--- split into snippet ---")
        df_reviews['snippet'] = df_reviews['text'].progress_apply(lambda x: sent_tokenize(x))
        df_reviews = df_reviews.explode('snippet').reset_index(drop=True)

        print("\n--- pre-lex filter ---")
        if pre_lex_filter:
            df_reviews = self.setentiment_pre_filtering(df_reviews)

        print("\n--- generating ngram filter ---")
        df_reviews['snippet_ngram'] = df_reviews['snippet'].progress_apply(lambda x: self.generate_ngrams(x, n_gram_size, n_gram_jump))
        df_reviews = df_reviews.explode('snippet_ngram').reset_index(drop=True)

        print("\n--- encoding sentences ---")
        df_reviews['embeddings'] = self.encode_sentences(df_reviews['snippet_ngram'].values)

        print("\n--- generate correlation sentences ---")
        correlated_snippets = self.generate_search(df_reviews, correlation_thold, pre_lex_filter, pos_NN_filter)

        if pos_NN_filter:
            print("\n--- generate sentiment NN pos filtering ---")
            correlated_snippets = self.setentiment_pos_filtering(correlated_snippets)
        return correlated_snippets

    def graph_explorer(self, detected_issues):

      pos = nx.spring_layout(self.issue_network) # options (iterations=600, seed=300) get coordinates of vertices for visualization
      for node in self.issue_network.nodes():
        self.issue_network.nodes[node]['pos'] = pos[node]

      ### EDGES
      edge_x = []
      edge_y = []

      # adding coordinates
      for edge in self.issue_network.edges():
          x0, y0 = self.issue_network.nodes[edge[0]]['pos']
          x1, y1 = self.issue_network.nodes[edge[1]]['pos']
          edge_x.append(x0)
          edge_x.append(x1)
          #edge_x.append(None)
          edge_y.append(y0)
          edge_y.append(y1)
          #edge_y.append(None)

      ### NODES
      node_x = []
      node_y = []

      # adicionando as coordenadas
      for node in self.issue_network.nodes():
          x, y = self.issue_network.nodes[node]['pos']
          node_x.append(x)
          node_y.append(y)

      #adding nodes id
      node_id = []
      for node in self.issue_network.nodes():
          node_id.append(node)

      # adding nodes text
      node_text = []
      for node in self.issue_network.nodes():
          node_text.append(self.issue_network.nodes[node]['issue'])
      # adding nodes degree
      node_degree = []
      for (node, val) in self.issue_network.degree():
          node_degree.append(val)
      # adding detected issues
      node_detected_issue = []
      node_detected_issue_tier = []
      for node in self.issue_network.nodes():
        issue = self.issue_network.nodes[node]['issue']
        if (detected_issues['issue_tier_1'].eq(issue.strip())).any():
          node_detected_issue.append(1)

        elif (detected_issues['issue_tier_2'].eq(issue.strip())).any():
          node_detected_issue.append(1)

        else:
          node_detected_issue.append(0)

      # adding edges
      edges = []
      for node in self.issue_network.nodes():
        for p in nx.edges(self.issue_network, node):
          edges.append(p)

      # Nodes dataframe
      # Nodes dataframe
      node_x = np.array(node_x)
      df_node_x = pd.DataFrame(node_x,columns=['x'])

      node_y = np.array(node_y)
      df_node_y = pd.DataFrame(node_y,columns=['y'])

      node_text = np.array(node_text)
      df_node_text = pd.DataFrame(node_text,columns=['text'])

      node_id = np.array(node_id)
      df_node_id = pd.DataFrame(node_id,columns=['id'])

      node_degree = np.array(node_degree)
      df_node_degree = pd.DataFrame(node_degree,columns=['degree'])

      node_detected_issue = np.array(node_detected_issue)
      df_node_detected_issue = pd.DataFrame(node_detected_issue,columns=['degree'])


      df_node = pd.DataFrame(columns=['x','y','text','id','degree','detected_issue'])
      df_node['x'] = df_node_x
      df_node['y'] = df_node_y
      df_node['text'] = df_node_text
      df_node['id'] = df_node_id
      df_node['degree'] = df_node_degree
      df_node['detected_issue'] = df_node_detected_issue
      # Edges dataframe
      edges = np.array(edges)
      df_edge = pd.DataFrame(edges,columns=['from','to'])

      return df_node, df_edge
