#!/usr/bin/env python3

import pprint
import time
import typing

import requests

class AppStoreReviews:

    country = 'us'
    app_id = ''
    page = 1

    def __init__(self, country, app_id,page):
        self.country = country
        self.app_id = app_id
        self.page = page

    def is_error_response(self,http_response, seconds_to_sleep: float = 1) -> bool:
        """
        Returns False if status_code is 503 (system unavailable) or 200 (success),
        otherwise it will return True (failed). This function should be used
        after calling the commands requests.post() and requests.get().

        :param http_response:
            The response object returned from requests.post or requests.get.
        :param seconds_to_sleep:
            The sleep time used if the status_code is 503. This is used to not
            overwhelm the service since it is unavailable.
        """
        if http_response.status_code == 503:
            time.sleep(seconds_to_sleep)
            return False

        return http_response.status_code != 200


    def get_json(self,url) -> typing.Union[dict, None]:
        """
        Returns json response if any. Returns None if no json found.

        :param url:
            The url go get the json from.
        """
        response = requests.get(url)
        if self.is_error_response(response):
            return None
        json_response = response.json()
        return json_response


    def get_reviews(self) -> typing.List[dict]:
        """
        Returns a list of dictionaries with each dictionary being one review.

        :param app_id:
            The app_id you are searching.
        :param page:
            The page id to start the loop. Once it reaches the final page + 1, the
            app will return a non valid json, thus it will exit with the current
            reviews.
        """
        _reviews: typing.List[dict] = [{}]

        while True:
            url = (f'https://itunes.apple.com/{self.country}/rss/customerreviews/page={self.page}/id={self.app_id}/sortBy=mostRecent/json')
            json = self.get_json(url)

            if not json:
                return _reviews

            data_feed = json.get('feed')

            try:
                if not data_feed.get('entry'):
                    self.get_reviews(self.app_id, self.page + 1)
                _reviews += [
                    {
                        'reviewId': int(entry.get('id').get('label')),
                        'date': entry.get('updated').get('label'),
                        'title': entry.get('title').get('label'),
                        'author': entry.get('author').get('name').get('label'),
                        'author_url': entry.get('author').get('uri').get('label'),
                        'version': entry.get('im:version').get('label'),
                        'rating': entry.get('im:rating').get('label'),
                        'content': entry.get('content').get('label'),
                        'vote_count': entry.get('im:voteCount').get('label'),
                        'page': self.page
                    }
                    for entry in data_feed.get('entry')
                    if not entry.get('im:name')
                ]
                self.page += 1
            except Exception:
                return _reviews
