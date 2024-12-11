import os
from datasets import load_dataset
from huggingface_hub import login
from CalQuest_PT.extractors.utils import get_first_user_content, get_first_bot_content, is_portuguese

def get_wc(): 
    '''
    TODO Documentação
    '''
    
    login(token=os.environ['TOKEN_HUGGINGFACE'])
    dataset = load_dataset("allenai/WildChat")
    
    sampled_dataset = dataset["train"].shuffle(seed=42).select(range(5))
    sampled_dataset = sampled_dataset.map(lambda x: {'first_user_content': get_first_user_content(x['conversation'])})
    sampled_dataset = sampled_dataset.map(lambda x: {'first_bot_content': get_first_bot_content(x['conversation'])})
    sampled_dataset = sampled_dataset.filter(lambda x: is_portuguese(x['first_user_content']), load_from_cache_file=False)
    final_dataset = sampled_dataset.remove_columns([col for col in sampled_dataset.column_names if col not in ['conversation_id', 'first_user_content', 'first_bot_content', 'first_bot_country']])
    final_dataset = final_dataset.to_pandas()
    final_dataset = final_dataset.rename(columns={'conversation_id': 'unid', 'first_user_content': 'question', 'first_bot_content': 'wc_answer', 'first_bot_country': 'country'})
    
    final_dataset['source'] = 'wildchat'
    
    return final_dataset[['unid', 'question', 'source']].copy()
