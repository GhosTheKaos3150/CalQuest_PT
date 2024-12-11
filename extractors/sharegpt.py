import os
import pandas as pd
from CalQuest_PT.extractors.utils import download_file, is_portuguese

def get_sg(output_folder_path):
    '''
    TODO Documentação
    '''


    if not os.path.exists(output_folder_path + '/sg_90k_part1.json') or not os.path.exists(output_folder_path + 'sg_90k_part2.json'):
        print("ShareGPT files not found, downloading")
        urls = [
            'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1.json',
            'https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2.json'
        ]
        for url in urls:
            download_file(url, output_folder_path + url.split('/')[-1])
    
    else:
        print("ShareGPT files found")
    
    df_sg_1 = pd.read_json(output_folder_path + "sg_90k_part1.json")
    df_sg_2 = pd.read_json(output_folder_path + "sg_90k_part2.json")
    df_sg = pd.concat([df_sg_1, df_sg_2], ignore_index=True)
    
    df_sg['query'] = df_sg['conversations'].apply(lambda x: next((item['value'] for item in x if item['from'] == 'human'), None))
    df_sg['answer'] = df_sg['conversations'].apply(lambda x: next((item['value'] for item in x if item['from'] == 'gpt'), None))
    
    df_sg['query'] = df_sg['query'].astype(str)
    df_sg['query'] = df_sg['query'].convert_dtypes()
    df_sg = df_sg[df_sg['query'].str.split().apply(len) > 1]
    
    df_sg['answer'] = df_sg['answer'].astype(str)
    df_sg['answer'] = df_sg['answer'].convert_dtypes()
    
    df_sg = df_sg.drop(columns=['conversations'])
    
    df_sg = df_sg.rename(columns={'query': 'sg_question', 'answer': 'sg_answer'})
    
    df_sg = df_sg[df_sg['sg_question'].apply(lambda x: is_portuguese(x))]
    df_sg['source'] = 'sharegpt'
    df_sg = df_sg.rename(columns={'id': 'unid', 'sg_question': 'question'})

    return df_sg[['unid', 'question', 'source']].copy()