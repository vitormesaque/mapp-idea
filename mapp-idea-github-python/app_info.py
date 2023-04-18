#!/usr/bin/env python3
from google_play_scraper import app
from itunes_app_scraper.scraper import AppStoreScraper
import pandas as pd
import numpy as np
import sys
import os
import pathlib

def main():

    dataset_name = sys.argv[1]
    ios_id_num = sys.argv[2]
    platform = sys.argv[3]

    outdir = './python/tmp/output-'+dataset_name+'/'
    #if not os.path.exists(outdir):

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    #os.mkdir(outdir)

    #try:
    if platform == 'android':
        info_app = app(
            dataset_name,
            lang='en', # defaults to 'en'
            country='us' # defaults to 'us'
        )
        df_app_info = pd.json_normalize(info_app)

    if platform == 'ios':
        scraper = AppStoreScraper()
        info_app = scraper.get_app_details(app_id=ios_id_num, country="us", lang="en")
        df_app_info = pd.json_normalize(info_app)
        df_app_info = df_app_info[['sellerName','trackName', 'description', 'currentVersionReleaseDate','artworkUrl100', 'averageUserRating','version', 'userRatingCount', 'trackViewUrl']].copy()
        df_app_info.columns = ['developer', 'title', 'description', 'released', 'icon', 'score','version', 'reviews', 'url']


    df_app_info = pd.json_normalize(info_app)
    df_app_info.to_csv(outdir + dataset_name + '_info.csv')

    print(outdir + dataset_name + '_info.csv')

    #except:
        #print('0')

if __name__ == "__main__":
    main()
