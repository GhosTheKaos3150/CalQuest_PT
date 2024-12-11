import os
import pandas as pd
import uuid
from apify_client import ApifyClient

def get_reddit():
    '''
    TODO Documentação
    '''
    
    reddit_data = download_from_apify()
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
    TODO Documentação
    '''
    
    if not os.path.exists("./data_gen/reddit_pt_br_lg.xlsx"):
        # Estimated cost: USD$ 20
        urls = [
            ('https://www.reddit.com/r/PergunteReddit/', 1000),
            ('https://www.reddit.com/r/brdev/', 1000),
            ('https://www.reddit.com/r/conselhodecarreira/', 1000),
            ('https://www.reddit.com/r/programacao/', 1000),
            ('https://www.reddit.com/r/askacademico/', 1000),
            ('https://www.reddit.com/r/ApoioVet/', 1000),
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
            
            dataset = dataset[['id', 'title', 'parsedCommunityName', 'isAd', 'over18', 'createdAt', 'scrapedAt']]
            
            dataset.dropna(subset=['id', 'title', 'parsedCommunityName'])
            
            dataset['id'] = dataset['id'].apply(lambda x: uuid.uuid4())
            
            dfs.append(dataset)
        
        final_df = pd.concat(dfs)
        final_df.to_excel("./data_gen/reddit_pt_br_lg.xlsx")
        
    else:
        final_df = pd.read_excel('./data_gen/reddit_pt_br_lg.xlsx')
        final_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    return final_df