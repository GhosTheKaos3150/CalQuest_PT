import re
import spacy
import pandas as pd
from cleantext import clean
from CalQuest_PT.inference import question_5w2h

def liguistic_metrics(df: pd.DataFrame, name='all'):
    '''
    TODO Documentação
    '''
    
    # 5W-2H Questions and metrics
    df = generate_5w2h(df, name)
    
    # Tokenization and metrics
    df = generate_tokenization(df, name)
    
    return df


def generate_tokenization(df: pd.DataFrame, name='all'):
    '''
    TODO Documentação
    '''
    
    metrics_tokenization = {
        "source": [],
        "vocab_size": [],
        "avg_words_per_sample": [],
        "type_token_ratio": []
    }
    
    if name == 'all':
        nlp = spacy.load("pt_core_news_sm")
        
        df['question_spacy'] = df['question'].apply(lambda x: clean(x, lower=True, no_punct=True))

        df["doc"] = df["question_spacy"].apply(lambda x: nlp(x))
        df['tokens'] = df["doc"].apply(lambda x: [token.text for token in x])
        df['tokens_join'] = df["doc"].apply(lambda x: ' '.join([token.text for token in x]))
        df["tokens_len"] = df["tokens"].apply(lambda x: len(x))
        df["tokens_len"].fillna(0, inplace=True)
        
        df['ttr'] = df.apply(lambda x: len(list(set(x['tokens_join'].split(" "))))/len(x['tokens_join'].split(" ")), axis=1)
        df['nwords'] = df.apply(lambda x: len(list(set(x['question'].split(" ")))), axis=1)
        df['nunique_tk'] = df.apply(lambda x: len(list(set(x['tokens_join'].split(" ")))), axis=1)
    
    metrics_tokenization['source'].append("all")
    metrics_tokenization['vocab_size'].append(len(list(set(' '.join(df['question_spacy']).split(" ")))))
    metrics_tokenization['avg_words_per_sample'].append(df['nwords'].mean())
    metrics_tokenization['type_token_ratio'].append(df['ttr'].mean())
    
    for source in ['reddit', 'wildchat', 'sharegpt']:
        metrics_tokenization['source'].append(source)
        metrics_tokenization['vocab_size'].append(len(list(set(' '.join(df.loc[df['source'] == source, 'question_spacy']).split(" ")))))
        metrics_tokenization['avg_words_per_sample'].append(df.loc[df['source'] == source, 'nwords'].mean())
        metrics_tokenization['type_token_ratio'].append(df.loc[df['source'] == source, 'ttr'].mean())
    
    metrics_tokenization = pd.DataFrame(metrics_tokenization)
    metrics_tokenization.to_excel(f'./metrics_gen/metrics_tokenization_{name}.xlsx')
    
    return df


def generate_5w2h(df: pd.DataFrame, name='all'):
    '''
    TODO Documentação
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
        
        df['5w2h'] = ''

        df['5w2h'] = df.apply(lambda x: 'O quê' if re_o_que.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        df['5w2h'] = df.apply(lambda x: 'Por quê' if re_por_que.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        df['5w2h'] = df.apply(lambda x: 'Quem' if re_quem.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        df['5w2h'] = df.apply(lambda x: 'Onde' if re_onde.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        df['5w2h'] = df.apply(lambda x: 'Quando' if re_quando.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        df['5w2h'] = df.apply(lambda x: 'Como' if re_como.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        df['5w2h'] = df.apply(lambda x: 'Quanto' if re_quanto.search(x['tokens_join']) and x['5w2h'] == '' else x['5w2h'], axis=1)
        
        df = question_5w2h.generate_inference(df)

    metrics_inference['source'].append('all')
    metrics_inference['what'].append(df[df['5w2h'] == 'O quê'].shape[0])
    metrics_inference['why'].append(df[df['5w2h'] == 'Por quê'].shape[0])
    metrics_inference['who'].append(df[df['5w2h'] == 'Quem'].shape[0])
    metrics_inference['where'].append(df[df['5w2h'] == 'Onde'].shape[0])
    metrics_inference['when'].append(df[df['5w2h'] == 'Quando'].shape[0])
    metrics_inference['how'].append(df[df['5w2h'] == 'Como'].shape[0])
    metrics_inference['how much'].append(df[df['5w2h'] == 'Quanto'].shape[0])
    metrics_inference['other'].append(df[df['5w2h'] != 'Outro'].shape[0])

    for source in ['reddit', 'wildchat', 'sharegpt']:
        metrics_inference['source'].append(source)
        metrics_inference['what'].append(df[(df['source'] == source) & (df['5w2h'] == 'O quê')].shape[0])
        metrics_inference['why'].append(df[(df['source'] == source) & (df['5w2h'] == 'Por quê')].shape[0])
        metrics_inference['who'].append(df[(df['source'] == source) & (df['5w2h'] == 'Quem')].shape[0])
        metrics_inference['where'].append(df[(df['source'] == source) & (df['5w2h'] == 'Onde')].shape[0])
        metrics_inference['when'].append(df[(df['source'] == source) & (df['5w2h'] == 'Quando')].shape[0])
        metrics_inference['how'].append(df[(df['source'] == source) & (df['5w2h'] == 'Como')].shape[0])
        metrics_inference['how much'].append(df[(df['source'] == source) & (df['5w2h'] == 'Quanto')].shape[0])
        metrics_inference['other'].append(df[(df['source'] == source) & (df['5w2h'] != 'Outro')].shape[0])
    
    metrics_inference = pd.DataFrame(metrics_inference)
    metrics_inference.to_excel(f'./metrics_gen/metrics_5w2h_{name}.xlsx')
    
    return df
    
    