import os
import pandas as pd
import uuid
from apify_client import ApifyClient

def get_reddit():
    '''
    This function will download and pre-process Reddit data. This function uses an
    Apify actor for Reddit data crawling, as using Reddit Data API is difficult. For using
    Apify, you will need a subscription and providing the API Key on the .env file.
    
    Parameters: - 
    
    Return: Reddit Data Processed
    '''
    
    # Download
    reddit_data = download_from_apify()
    
    # Pre-Process
    questions = reddit_data['title'].apply(lambda x: 
                str(x).lower().__contains__("?") or str(x).lower().__contains__("por que") or str(x).lower().__contains__("por quê") or
                str(x).lower().__contains__("o que") or str(x).lower().__contains__("quem") or str(x).lower().__contains__("como") or
                str(x).lower().__contains__("onde") or str(x).lower().__contains__("quando") or str(x).lower().__contains__("qual") or
                str(x).lower().__contains__("quais") or str(x).lower().__contains__("que"))
    
    if not 'is_question' in reddit_data.columns:
        reddit_data['is_question'] = questions
        reddit_data.to_excel("./data_gen/reddit_pt_br_lg.xlsx")
    
    if reddit_data['isAd'].any() or reddit_data['over18'].any():
        reddit_data = reddit_data[~reddit_data['isAd'] & ~reddit_data['over18']]
        reddit_data.to_excel("./data_gen/reddit_pt_br_lg.xlsx")
        
    reddit_data.rename(columns={'id': 'unid',  'title': 'question'})
    reddit_data['source'] = 'reddit'
        
    return reddit_data[['unid', 'question', 'source']].copy()

def download_from_apify():
    '''
    This function downloads Reddit Data, using an Apify actor for data
    acquisition. This function uses an Apify actor for Reddit data crawling, as using Reddit Data API 
    is difficult. For using Apify, you will need a subscription and providing the API Key on the .env 
    file. You may need an Apify Subscription for using the actor, since the platform free-tier will not
    be enough for crawling the original 5k results.
    
    NOTE: Reddit data will vary over time, since the forums will be updated daily by Reddit users. 
    Executing the process in the future will result on different Reddit data being processed. If you are
    trying to reproduce the article results, copy "./experiment_data/reddit_pt_br_lg.xlsx" to "./data_gen/" 
    folder.
    
    Parameters: -
    
    Return: Reddit raw data.
    '''
    
    if not os.path.exists("./data_gen/reddit_pt_br_lg.xlsx"):
        # Estimated cost for 1k results per subreddit: USD$ 20
        n_results = 1000
        
        urls = [
            ('https://www.reddit.com/r/PergunteReddit/', n_results),
            ('https://www.reddit.com/r/brdev/', n_results),
            ('https://www.reddit.com/r/conselhodecarreira/', n_results),
            ('https://www.reddit.com/r/programacao/', n_results),
            ('https://www.reddit.com/r/askacademico/', n_results),
            ('https://www.reddit.com/r/ApoioVet/', n_results),
        ]
        
        dfs = []
        
        client = ApifyClient(os.environ['API_API_KEY'])
        for url in urls:
            run_input = {
                "startUrls": [{ "url": url[0] }],
                "skipComments": True,
                "skipUserPosts": False,
                "skipCommunity": True,
                "searchPosts": True,
                "searchComments": False,
                "searchCommunities": False,
                "searchUsers": False,
                "sort": "new",
                "includeNSFW": False,
                "maxItems": url[1],
                "maxPostCount": url[1],
                "scrollTimeout": 40,
                "proxy": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"],
                },
                "debugMode": False,
            }
            
            run = client.actor("oAuCIx3ItNrs2okjQ").call(run_input=run_input)
            
            pre_data_records = []
            for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                pre_data_records.append(item)
                
            dataset = pd.DataFrame(pre_data_records)
            
            dataset = dataset[['title', 'isAd', 'over18', 'scrapedAt']]
            
            dataset.dropna(subset=['title'])
            
            dataset['id'] = dataset['title'].apply(lambda x: uuid.uuid4())
            
            dfs.append(dataset)
        
        final_df = pd.concat(dfs)
        final_df.to_excel("./data_gen/reddit_pt_br_lg.xlsx")
        
    else:
        final_df = pd.read_excel('./data_gen/reddit_pt_br_lg.xlsx')
        final_df.drop(columns=['Unnamed: 0'], inplace=True)
        
        if not "id" in final_df.columns:
            final_df['id'] = final_df['title'].apply(lambda x: uuid.uuid4())
    
    return final_df