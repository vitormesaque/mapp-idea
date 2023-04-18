#!/usr/bin/env python3
from google_play_scraper import Sort, reviews, reviews_all, app
from itunes_app_scraper.scraper import AppStoreScraper
from app_store_scraper import AppStore
from dateutil.relativedelta import relativedelta
from sklearn.cluster import MiniBatchKMeans
from datetime import date
from datetime import datetime
from tqdm import tqdm
from math import ceil
import issue_detector as detector_pkg
import app_store_reviews as app_store_pkg
import pandas as pd
import numpy as np
import requests
import pathlib
import shutil
import sys
import os
import logging

pd.set_option('display.max_colwidth', None)

def main ():

    outdir = sys.argv[1]
    dataset_name = sys.argv[2]
    confidence_min = float(sys.argv[3]) #0.6
    severity_min = int(sys.argv[4]) #5
    max_number_reviews = int(sys.argv[5]) #5
    ios_app_name = sys.argv[6] #5
    ios_app_id = sys.argv[7] #5
    platform = sys.argv[8] #5
    lang = sys.argv[9] #5
    country = sys.argv[10] #5

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    log = detector_pkg.Log(outdir)
    log.write_status('_INI_')

    df_issues = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'model', 'issues', 'issues.pkl'))
    df_issues.fillna('',inplace=True)

    SYNC = False
    last_review_sync = False
    file_exists = os.path.exists(outdir + dataset_name + '_last_sync.csv')
    if file_exists:
        last_sync = pd.read_csv(outdir + dataset_name + '_last_sync.csv',index_col=0)
        last_review_sync = last_sync.reviewId
        SYNC = True
        log.write_status('_SYNC_')

    #now we will Create and configure logger
    logging.basicConfig(filename=outdir + 'sys_output_console.log',level=logging.INFO,
    					format='%(asctime)s %(message)s',
    					filemode='w')

    #Let us Create an object
    logger=logging.getLogger()

    log.write_step(1)
    logger.info("downloading reviews")

    if platform == 'android':
        info_app = app(
            dataset_name,
            lang=lang, # defaults to 'en'
            country=country # defaults to 'us'
        )
        df_app_info = pd.json_normalize(info_app)

    if platform == 'ios':
        scraper = AppStoreScraper()
        info_app = scraper.get_app_details(app_id=ios_app_id, country=country, lang=lang)
        df_app_info = pd.json_normalize(info_app)
        df_app_info = df_app_info[['sellerName','trackName', 'description', 'currentVersionReleaseDate','artworkUrl100', 'averageUserRating','version', 'userRatingCount', 'trackViewUrl']].copy()
        df_app_info.columns = ['developer', 'title', 'description', 'released', 'icon', 'score','version', 'reviews', 'url']


    df = scrapper (platform,dataset_name, ios_app_id, lang, country, max_number_reviews)

    df.to_csv(outdir + 'df.csv')

    if SYNC:
        last_row_sync = df[(df['reviewId'].isin(last_review_sync))]
        try:
            print('-- last_row_sync found -- ')
            print(last_row_sync.index.tolist()[0])
            df = df.head(last_row_sync.index.tolist()[0])
        except IndexError:
            print('-- not found last_row_sync in new dataframe -- ')
            #log.write_status('_DONE_') #already updated
            #sys.exit()

        df_last_r_index = pd.read_csv(outdir + dataset_name + '_reviews.csv',index_col=0) #load current reviews issues
        df_last_r_index.reset_index(drop=True, inplace=True) #reset index
        df_last_r_index = df_last_r_index.loc[:, ~df_last_r_index.columns.str.contains('^Unnamed')] #remove header trash
        df_last_i_index = pd.read_csv(outdir + dataset_name + '_issues.csv',index_col=0)

        df_last_r_index['r_index'] = df_last_r_index.apply(lambda row : row['r_index']+len(df), axis=1) #sync index
        df_last_i_index['r_index'] = df_last_i_index.apply(lambda row : row['r_index']+len(df), axis=1) #sync index

        df_last_r_index.to_csv(outdir + dataset_name + '_reviews.csv')
        df_last_i_index.to_csv(outdir + dataset_name + '_issues.csv', index=True)


    if len(df) > 1:

        print('-- len(df): ' + str(len(df)) + ' -- ')
        last_sync = df.head(1)

        if platform == 'android':
            df = df[['content', 'score', 'at']].copy()
        if platform == 'ios':
            #df = df[['review', 'rating', 'date']].copy()
            df = df[['content', 'rating', 'date']].copy()
        df.columns = ['content', 'rating', 'date']
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
        df['date'] = df['date'].dt.strftime("%Y-%m-%d")
        df = df[df['content'].notna()]
        df = df[df['content'].str.split().str.len() > 1]
        df = df.reset_index()
        df['r_index'] = df.index
        df['y_true'] = 'Non Issue'
        df.columns = ['index', 'text', 'rating', 'date', 'r_index', 'y_true']

        today = datetime.now()
        total_num_raw_reviews = pd.DataFrame([[today, len(df)]], columns=['date', 'total'])
        if SYNC:
            tmp_total_num_raw_reviews = pd.read_csv(outdir + dataset_name + '_total_processed.csv',index_col=0)
            total_num_raw_reviews = pd.concat([tmp_total_num_raw_reviews, total_num_raw_reviews], ignore_index=True, sort=False)

        input =  {
            "pre_lex_filter": True,
            "pos_NN_filter": True,
            "correlation_thold": 0.8,
            "n_gram_size": 7,
            "n_gram_jump": 5,
            "reviews": df
        }

        log.write_step(2)
        logger.info("loading model")

        model_runner = detector_pkg.IssueDetector()

        log.write_step(3)
        logger.info("detecting issues")

        print("detecting issues")

        predictions = predict(input, model_runner)
        df_pred = format_output(predictions, df)

        if len(df_pred): #any()

            issues = df_pred.copy()

            issues_filter = issues[['r_index', 'text', 'date', 'rating', 'entity', 'issue', 'issue_tier_1', 'issue_tier_2', 'snippet_sentiment', 'entity_sentiment_confidence', 'severity']].copy()
            issues_filter['y_pred'] = 'Issue Detected'
            issues_filter.columns = ['r_index', 'review', 'date', 'rating', 'sentence', 'issue', 'issue_tier_1', 'issue_tier_2', 'score', 'confidence', 'severity', 'y_pred']


            #issues_filter = issues[issues.confidence >= confidence_min]
            #issues_filter = issues_filter[issues_filter.severity >= severity_min]

            log.write_step(4)
            logger.info("creating issue network")

            # Graph generator
            df_node, df_edge = model_runner.graph_explorer(detected_issues=issues_filter)

            log.write_step(5)
            logger.info("saving the output data")

            # Detected Issue output

            issues_filter['score'] = issues_filter['score'].abs()

            severity = (issues_filter.groupby(['r_index', 'issue', 'severity'], as_index=False).mean(numeric_only=True).groupby('r_index')['severity'].agg('max'))
            score = (issues_filter.groupby(['r_index', 'issue'], as_index=False).mean(numeric_only=True).groupby('r_index')['score'].agg('max'))
            rating = (issues_filter.groupby('r_index')['rating'].agg('max'))
            date = (issues_filter.groupby('r_index')['date'].agg('max'))
            grouped_issue = issues_filter.groupby(['r_index'])

            grouped_lists = grouped_issue['issue'].apply(list)
            df_issue_grouped = pd.DataFrame(grouped_lists)
            df_issue_grouped['severity_mean'] = severity
            df_issue_grouped['score_max'] = score
            df_issue_grouped['rating'] = rating
            df_issue_grouped['date'] = date
            df_issue_grouped['y_pred'] = 'Detected Issue'


            # Normalization
            issue_scaled = df_issue_grouped.copy()
            max_value = df_issue_grouped['severity_mean'].max()
            min_value = df_issue_grouped['severity_mean'].min()
            issue_scaled['severity_mean'] = (df_issue_grouped['severity_mean'] - min_value) / (max_value - min_value)
            max_value = df_issue_grouped['score_max'].max()
            min_value = df_issue_grouped['score_max'].min()
            issue_scaled['score_max'] = (df_issue_grouped['score_max'] - min_value) / (max_value - min_value)
            issue_scaled = pd.DataFrame(issue_scaled)
            pd.merge(issue_scaled, issues_filter, on='r_index')


            # Normalization of severity (priority) in detected issues
            max_value = issues_filter['severity'].max()
            min_value = issues_filter['severity'].min()
            issues_filter['severity'] = (issues_filter['severity'] - min_value) / (max_value - min_value)

            #saving r_index in new column
            issue_scaled.index.name = None
            issue_scaled['r_index'] = issue_scaled.index

            #restart index
            issues_filter.reset_index(drop=True, inplace=True)
            issue_scaled.reset_index(drop=True, inplace=True)


            #saving data
            if SYNC:
                last_df_node = pd.read_csv(outdir +dataset_name + '_node_network.csv',index_col=0)
                df_node['detected_issue'] = df_node.loc[df_node.text.isin(last_df_node.text), ['detected_issue']] = last_df_node[['detected_issue']]

                tmp_issues_filter = pd.read_csv(outdir +dataset_name + '_reviews.csv',index_col=0)
                tmp_issue_scaled = pd.read_csv(outdir +dataset_name + '_issues.csv',index_col=0)

                new_issues_filter = pd.concat([tmp_issues_filter, issues_filter], ignore_index=True, sort=False)
                new_issue_scaled = pd.concat([tmp_issue_scaled, issue_scaled], ignore_index=True, sort=False)

                new_issues_filter["r_index"] = pd.to_numeric(new_issues_filter["r_index"])
                new_issue_scaled["r_index"] = pd.to_numeric(new_issue_scaled["r_index"])
                new_issues_filter.sort_values(by=['r_index'], ascending=True, inplace=True)
                new_issue_scaled.sort_values(by=['r_index'], ascending=True, inplace=True)

                new_issues_filter.reset_index(drop=True, inplace=True)
                new_issue_scaled.reset_index(drop=True, inplace=True)

                new_issues_filter.to_csv(outdir + dataset_name + '_reviews.csv')
                new_issue_scaled.to_csv(outdir + dataset_name + '_issues.csv')
            else:
                df_edge.to_csv(outdir +dataset_name + '_edge_network.csv')
                issues_filter.to_csv(outdir +dataset_name + '_reviews.csv')
                issue_scaled.to_csv(outdir +dataset_name + '_issues.csv')

            df_node.to_csv(outdir +dataset_name + '_node_network.csv')
            df_app_info.to_csv(outdir +dataset_name + '_info.csv')
            last_sync.to_csv(outdir +dataset_name + '_last_sync.csv')
            total_num_raw_reviews.to_csv(outdir + dataset_name + '_total_processed.csv')

    log.write_step(6)
    log.write_status('_DONE_')


def format_output(output, reviews_list):
  cp_reviews_list = reviews_list.copy()

  df_columns = ['r_index','text', 'date', 'rating','entity', 'issue', 'issue_dist', 'issue_tier', 'issue_tier_2', 'issue_tier_2_dist', 'issue_tier_1', 'issue_tier_1_dist', 'severity', 'snippet_sentiment', 'entity_sentiment', 'entity_sentiment_confidence']
  dict_to_df = {cur_col: [] for cur_col in df_columns}

  #processed = set()
  for cur_output in output:
    #processed.add(cur_output['text'])
    for cur_entity in cur_output['entities']:
      dict_to_df['r_index'].append(cur_output['r_index'])
      dict_to_df['text'].append(cur_output['text'])
      dict_to_df['date'].append(cur_output['date'])
      dict_to_df['rating'].append(cur_output['rating'])
      for cur_col in cur_entity.keys():
        dict_to_df[cur_col].append(cur_entity[cur_col])

  return pd.DataFrame(dict_to_df)

def predict(input, model_runner):
    batch_size = 4096


    pre_lex_filter = input['pre_lex_filter'] # True
    pos_NN_filter = input['pos_NN_filter'] # True
    correlation_thold = input['correlation_thold'] # 0.8
    n_gram_size = input['n_gram_size'] # 7
    n_gram_jump = input['n_gram_jump'] # 5

    chunk_n = ceil(len(input["reviews"]) / batch_size)
    inter = np.array_split(input["reviews"], chunk_n)
    response = []

    for chunk in inter:
        #model_input = [x["text"] for x in chunk]
        model_input = chunk
        predictions = model_runner.predict(model_input, correlation_thold, n_gram_size, n_gram_jump, pre_lex_filter, pos_NN_filter)
        for cur_r_index in predictions:

            row = model_input[model_input.r_index == cur_r_index]
            response.append({
                "r_index" : row['r_index'].to_string(index=False),
                "text" : row['text'].to_string(index=False),
                "date" : row['date'].to_string(index=False),
                "rating" : row['rating'].to_string(index=False),
                "entities" : predictions[cur_r_index]
            })

    return response

def scrapper (platform, dataset_name, ios_app_id, lang, country, max_number_reviews):

    if platform == 'android' :
        result, continuation_token = reviews(
            dataset_name,
            lang=lang, # defaults to 'en'
            country=country, # defaults to 'us'
            sort=Sort.NEWEST, # defaults to Sort.NEWEST
            count=max_number_reviews, # defaults to 100
            filter_score_with=None # defaults to None(means all score)
        )

        df = pd.DataFrame(np.array(result),columns=['review'])
        df = df.join(pd.DataFrame(df.pop('review').tolist()))

    if platform == 'ios':
        # Method 1 (sort by date is not supported)
        #result = AppStore(country=country, app_name='', app_id=ios_app_id)
        #result.review(max_number_reviews)
        #df = pd.DataFrame(np.array(result.reviews),columns=['review'])
        #df = df.join(pd.DataFrame(df.pop('review').tolist()))

        # Method 2
        scraper_reviews = app_store_pkg.AppStoreReviews(country=country, app_id=ios_app_id,page=1)
        result = scraper_reviews.get_reviews()
        df = pd.DataFrame.from_dict(result)
        df = df.drop(0)



    return df


if __name__ == "__main__":

    main()
