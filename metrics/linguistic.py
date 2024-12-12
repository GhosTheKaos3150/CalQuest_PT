import re
import spacy
import pandas as pd
from cleantext import clean
from CalQuest_PT.inference import question_5w2h

def liguistic_metrics(dataset: pd.DataFrame, name='all'):
    '''
    Generates all linguistic metrics, both 5W-2H classification metrics and tokenization metrics.
    
    Parameters:
    : dataset - Dataset to be used for generating metrics (Full dataset or Golden Collection before inference)
    : name - Name of the source to be used.
    
    Return: Dataset with new 5W-2H and Tokens columns
    '''
    
    # 5W-2H Questions and metrics
    dataset = generate_5w2h(dataset, name)
    
    # Tokenization and metrics
    dataset = generate_tokenization(dataset, name)
    
    return dataset


def generate_tokenization(dataset: pd.DataFrame, name='all'):
    '''
    Generate tokenization columns and calculate matrics for them. Metrics will be exported to a file at the end.
    
    Parameters:
    : dataset - Dataset to be used for generating metrics (Full dataset or Golden Collection before inference)
    : name - Name of the source to be used.
    
    Return: Dataset modified with new columns.
    '''
    
    metrics_tokenization = {
        "source": [],
        "vocab_size": [],
        "avg_words_per_sample": [],
        "type_token_ratio": []
    }
    
    if name == 'all':
        nlp = spacy.load("pt_core_news_sm")
        
        dataset['question_spacy'] = dataset['question'].apply(lambda x: clean(x, lower=True, no_punct=True))

        dataset["doc"] = dataset["question_spacy"].apply(lambda x: nlp(x))
        dataset['tokens'] = dataset["doc"].apply(lambda x: [token.text for token in x])
        dataset['tokens_join'] = dataset["doc"].apply(lambda x: ' '.join([token.text for token in x]))
        dataset["tokens_len"] = dataset["tokens"].apply(lambda x: len(x))
        dataset["tokens_len"].fillna(0, inplace=True)
        
        dataset['ttr'] = dataset.apply(lambda x: len(list(set(x['tokens_join'].split(" "))))/len(x['tokens_join'].split(" ")), axis=1)
        dataset['nwords'] = dataset.apply(lambda x: len(list(set(x['question'].split(" ")))), axis=1)
        dataset['nunique_tk'] = dataset.apply(lambda x: len(list(set(x['tokens_join'].split(" ")))), axis=1)
    
    metrics_tokenization['source'].append("all")
    metrics_tokenization['vocab_size'].append(len(list(set(' '.join(dataset['question_spacy']).split(" ")))))
    metrics_tokenization['avg_words_per_sample'].append(dataset['nwords'].mean())
    metrics_tokenization['type_token_ratio'].append(dataset['ttr'].mean())
    
    for source in ['reddit', 'wildchat', 'sharegpt']:
        metrics_tokenization['source'].append(source)
        metrics_tokenization['vocab_size'].append(len(list(set(' '.join(dataset.loc[dataset['source'] == source, 'question_spacy']).split(" ")))))
        metrics_tokenization['avg_words_per_sample'].append(dataset.loc[dataset['source'] == source, 'nwords'].mean())
        metrics_tokenization['type_token_ratio'].append(dataset.loc[dataset['source'] == source, 'ttr'].mean())
    
    metrics_tokenization = pd.DataFrame(metrics_tokenization)
    metrics_tokenization.to_excel(f'./metrics_gen/metrics_tokenization_{name}.xlsx')
    
    return dataset


def generate_5w2h(dataset: pd.DataFrame, name='all'):
    '''
    Generate 5W-2H column and calculate matrics. Metrics will be exported to a file at the end.
    Descricao
    
    Parameters:
    : dataset - Dataset to be used for generating metrics (Full dataset or Golden Collection before inference)
    : name - Name of the source to be used.
    
    Return: Dataset modified with new column.
    '''

    metrics_inference = {
        "source": [],
        "what": [],
        "why": [],
        "who": [],
        "where": [],
        "when": [],
        "what": [],
        "what": [],
        "other": [],
    }
    
    if name == 'all':
        re_o_que = re.compile('o que|qual|quais|tem alguma')
        re_por_que = re.compile('por que|voce acredita|voces acreditam|vcs acreditam|voce sente|voces sentem|vcs sentem|voce acha|voces acham|vcs acham')
        re_quem = re.compile('quem')
        re_onde = re.compile('onde')
        re_quando = re.compile('quando')
        re_como = re.compile('como')
        re_quanto = re.compile('quanto')
        
        dataset['5w2h'] = ''

        dataset['5w2h'] = dataset.apply(lambda x: 'O quê' if re_o_que.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        dataset['5w2h'] = dataset.apply(lambda x: 'Por quê' if re_por_que.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        dataset['5w2h'] = dataset.apply(lambda x: 'Quem' if re_quem.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        dataset['5w2h'] = dataset.apply(lambda x: 'Onde' if re_onde.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        dataset['5w2h'] = dataset.apply(lambda x: 'Quando' if re_quando.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        dataset['5w2h'] = dataset.apply(lambda x: 'Como' if re_como.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        dataset['5w2h'] = dataset.apply(lambda x: 'Quanto' if re_quanto.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        
        dataset = question_5w2h.generate_inference(dataset)

    metrics_inference['source'].append('all')
    metrics_inference['what'].append(dataset[dataset['5w2h'] == 'O quê'].shape[0])
    metrics_inference['why'].append(dataset[dataset['5w2h'] == 'Por quê'].shape[0])
    metrics_inference['who'].append(dataset[dataset['5w2h'] == 'Quem'].shape[0])
    metrics_inference['where'].append(dataset[dataset['5w2h'] == 'Onde'].shape[0])
    metrics_inference['when'].append(dataset[dataset['5w2h'] == 'Quando'].shape[0])
    metrics_inference['how'].append(dataset[dataset['5w2h'] == 'Como'].shape[0])
    metrics_inference['how much'].append(dataset[dataset['5w2h'] == 'Quanto'].shape[0])
    metrics_inference['other'].append(dataset[dataset['5w2h'] != 'Outro'].shape[0])

    for source in ['reddit', 'wildchat', 'sharegpt']:
        metrics_inference['source'].append(source)
        metrics_inference['what'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'O quê')].shape[0])
        metrics_inference['why'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'Por quê')].shape[0])
        metrics_inference['who'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'Quem')].shape[0])
        metrics_inference['where'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'Onde')].shape[0])
        metrics_inference['when'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'Quando')].shape[0])
        metrics_inference['how'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'Como')].shape[0])
        metrics_inference['how much'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] == 'Quanto')].shape[0])
        metrics_inference['other'].append(dataset[(dataset['source'] == source) & (dataset['5w2h'] != 'Outro')].shape[0])
    
    metrics_inference = pd.DataFrame(metrics_inference)
    metrics_inference.to_excel(f'./metrics_gen/metrics_5w2h_{name}.xlsx')
    
    return dataset
    
    