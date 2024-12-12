import pandas as pd

from CalQuest_PT.extractors import reddit, sharegpt, wildchat
from CalQuest_PT.inference import inference_fs, inference_cot
from CalQuest_PT.metrics import linguistic, inference

import dotenv

def run():
    '''
    This function runs the process to compile the CalQuest.PT dataset and it's Golden Collection
    (Collection of 600 questions used for tests, metrics and comparisons).
    
    NOTE: Reddit data will vary over time, since the forums will be updated daily by Reddit users. 
    Executing the process in the future will result on different Reddit data being processed. If you are
    trying to reproduce the article results, copy "./experiment_data/reddit_pt_br_lg.xlsx" to "./data_gen/" folder.
    
    Parameters: -
    
    Return: -
    '''
    
    # Columns that begins with "hr_" are for Human Annotators 
    df_united = pd.DataFrame({
        'unid': [], 'question': [], 'source': [], 
        'hr_is_causal': [],  'hr_action_class': [], 'hr_pearl_class': []
    })
    
    df_reddit = reddit.get_reddit()
    df_wildchat = wildchat.get_wc()
    df_sharegpt = sharegpt.get_sg()
    
    df_united = pd.concat([df_united, df_reddit])
    df_united = pd.concat([df_united, df_wildchat])
    df_united = pd.concat([df_united, df_sharegpt])
    
    df_united.to_excel('./data_gen/unified_unmodfied.xlsx')
    
    # Linguistic Metrics
    df_united = linguistic.liguistic_metrics(df_united)
    df_united.to_excel('./data_gen/unified_modified_5w2h_tokenized.xlsx')
    
    # GPT Inference
    samples = { 'reddit': None, 'wildchat': None, 'sharegpt': None}
    
    for source in samples.keys():
        samples[source] = df_united[(df_united['source'] == source)].sample(200, random_state=42)
    
    golden_collection = pd.concat([samples[source] for source in samples.keys()])
    golden_collection = linguistic.liguistic_metrics(golden_collection, 'gold')
    golden_collection.to_excel("./data_gen/golden_collection.xlsx")
    
    gc_infered_few_shot = inference_fs.generate_inference(golden_collection.copy())
    gc_infered_chain_of_thought = inference_cot.generate_inference(golden_collection.copy())
    
    # Inference Metrics
    inference.generate_inference_metrics(gc_infered_few_shot)
    inference.generate_inference_metrics(gc_infered_chain_of_thought)


if __name__ == '__main__':
    # Loads all .env variables
    dotenv.load_dotenv()
    run()