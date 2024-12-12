import pandas as pd

def generate_inference_metrics(golden_collection: pd.DataFrame):
    '''
    This function generate Precision, Recall and F1-Score for the Golden Collection data.
    It will generate the metrics for all dataset and separated by source.
    
    Parameters:
    : golden_collection - Golden Collection Dataset, after inference.
    
    Return: -
    '''
    
    # Total
    all_precision = calculate_precision(golden_collection)
    all_recall = calculate_recall(golden_collection)
    all_f1_score = calculate_f1_score(all_precision, all_recall)
    
    golden_collection_metrics = pd.concat([all_precision, all_recall, all_f1_score])
    
    # By Source
    for source in ['reddit', 'wildchat', 'sharegpt']:
        source_precision = calculate_precision(golden_collection[golden_collection['source'] == source], source)
        source_recall = calculate_recall(golden_collection[golden_collection['source'] == source], source)
        source_f1_score = calculate_f1_score(all_precision, all_recall, source)
        
        golden_collection_metrics = pd.concat([golden_collection_metrics, source_precision, source_recall, source_f1_score])
    
    golden_collection_metrics.sort_values(by='metric')
    golden_collection_metrics.to_excel('./metrics_gen/golden_collect_metrics.xlsx')

def calculate_precision(golden_collection: pd.DataFrame, source='all') -> pd.Series:
    '''
    This function generates Precision for the Golden Collection. If no source is provided, will use all
    data.
    
    Parameters:
    : golden_collection - Golden Collection Dataset, after inference.
    : source - Data source used.
    
    Return: Series with metrics for all classes.
    '''
    
    precisions = {
        "source": source,
        "metric": 'Precision',
        "causality": 0.0,
        "non_causality": 0.0,
        "action_class_SCause": 0.0,
        "action_class_SEffect": 0.0,
        "action_class_SRelation": 0.0,
        "action_class_SSteps": 0.0,
        "action_class_SRecomm": 0.0,
        "pearl_class_Assocc": 0.0,
        "pearl_class_Interv": 0.0,
        "pearl_class_Contra": 0.0,
    }
    
    # Causality
    dividend = golden_collection[(golden_collection['hr_is_causal'] == True) & (golden_collection['hr_is_causal'] == golden_collection['is_causal'])].shape[0]
    divisor = golden_collection[(golden_collection['is_causal'] == True)].shape[0]
    
    precisions['causality'] = round(dividend/divisor, 2)
    
    # Non Causality
    dividend = golden_collection[(golden_collection['hr_is_causal'] == False) & (golden_collection['hr_is_causal'] == golden_collection['is_causal'])].shape[0]
    divisor = golden_collection[(golden_collection['is_causal'] == False)].shape[0]
    
    precisions['non_causality'] = round(dividend/divisor, 2)

    # Action Class (Seek-Cause)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Causa') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['action_class'] == 'Busca-Causa')].shape[0]
    
    precisions['action_class_SCause'] = round(dividend/divisor, 2)

    # Action Class (Seek-Effect)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Efeito') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['action_class'] == 'Busca-Efeito')].shape[0]

    precisions['action_class_SEffect'] = round(dividend/divisor, 2)

    # Action Class (Seek-Relation)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Relação') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['action_class'] == 'Busca-Relação')].shape[0]
    
    precisions['action_class_SRelation'] = round(dividend/divisor, 2)

    # Action Class (Seek-Steps)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Passos') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['action_class'] == 'Busca-Passos')].shape[0]
    
    precisions['action_class_SSteps'] = round(dividend/divisor, 2)

    # Action Class (Seek-Recomm)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Recomendação') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['action_class'] == 'Busca-Recomendação')].shape[0]
    
    precisions['action_class_SRecomm'] = round(dividend/divisor, 2)

    # Pearl Class (Assocciative)
    dividend = golden_collection[(golden_collection['hr_pearl_class'] == 'Associacional') & (golden_collection['hr_pearl_class'] == golden_collection['pearl_class'])].shape[0]
    divisor = golden_collection[(golden_collection['pearl_class'] == 'Associacional')].shape[0]
    
    precisions['pearl_class_Assocc'] = round(dividend/divisor, 2)

    # Pearl Class (Intervencionist)
    dividend = golden_collection[(golden_collection['hr_pearl_class'] == 'Intervencional') & (golden_collection['hr_pearl_class'] == golden_collection['pearl_class'])].shape[0]
    divisor = golden_collection[(golden_collection['pearl_class'] == 'Intervencional')].shape[0]
    
    precisions['pearl_class_Interv'] = round(dividend/divisor, 2)

    # Pearl Class (Contrafactual)
    dividend = golden_collection[(golden_collection['hr_pearl_class'] == 'Contrafactual') & (golden_collection['hr_pearl_class'] == golden_collection['pearl_class'])].shape[0]
    divisor = golden_collection[(golden_collection['pearl_class'] == 'Contrafactual')].shape[0]
    
    precisions['pearl_class_Contra'] = round(dividend/divisor, 2)
    
    return pd.Series(precisions)

def calculate_recall(golden_collection: pd.DataFrame, source='all') -> pd.Series:
    '''
    This function generates Recall for the Golden Collection. If no source is provided, will use all
    data.
    
    Parameters:
    : golden_collection - Golden Collection Dataset, after inference.
    : source - Data source used.
    
    Return: Series with metrics for all classes.
    '''
    
    recall = {
        "source": source,
        "metric": 'Recall',
        "causality": 0.0,
        "non_causality": 0.0,
        "action_class_SCause": 0.0,
        "action_class_SEffect": 0.0,
        "action_class_SRelation": 0.0,
        "action_class_SSteps": 0.0,
        "action_class_SRecomm": 0.0,
        "pearl_class_Assocc": 0.0,
        "pearl_class_Interv": 0.0,
        "pearl_class_Contra": 0.0,
    }
    
    # Causality
    dividend = golden_collection[(golden_collection['hr_is_causal'] == True) & (golden_collection['hr_is_causal'] == golden_collection['is_causal'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_is_causal'] == True)].shape[0]
    
    recall['causality'] = round(dividend/divisor, 2)
    
    # Non Causality
    dividend = golden_collection[(golden_collection['hr_is_causal'] == False) & (golden_collection['hr_is_causal'] == golden_collection['is_causal'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_is_causal'] == False)].shape[0]
    
    recall['non_causality'] = round(dividend/divisor, 2)

    # Action Class (Seek-Cause)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Causa') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Causa')].shape[0]
    
    recall['action_class_SCause'] = round(dividend/divisor, 2)

    # Action Class (Seek-Effect)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Efeito') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Efeito')].shape[0]
    
    recall['action_class_SEffect'] = round(dividend/divisor, 2)

    # Action Class (Seek-Relation)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Relação') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Relação')].shape[0]
    
    recall['action_class_SRelation'] = round(dividend/divisor, 2)

    # Action Class (Seek-Steps)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Passos') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Passos')].shape[0]
    
    recall['action_class_SSteps'] = round(dividend/divisor, 2)

    # Action Class (Seek-Recomm)
    dividend = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Recomendação') & (golden_collection['hr_action_class'] == golden_collection['action_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_action_class'] == 'Busca-Recomendação')].shape[0]
    
    recall['action_class_SRecomm'] = round(dividend/divisor, 2)

    # Pearl Class (Assocciative)
    dividend = golden_collection[(golden_collection['hr_pearl_class'] == 'Associacional') & (golden_collection['hr_pearl_class'] == golden_collection['pearl_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_pearl_class'] == 'Associacional')].shape[0]
    
    recall['pearl_class_Assocc'] = round(dividend/divisor, 2)

    # Pearl Class (Intervencionist)
    dividend = golden_collection[(golden_collection['hr_pearl_class'] == 'Intervencional') & (golden_collection['hr_pearl_class'] == golden_collection['pearl_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_pearl_class'] == 'Intervencional')].shape[0]
    
    recall['pearl_class_Interv'] = round(dividend/divisor, 2)

    # Pearl Class (Contrafactual)
    dividend = golden_collection[(golden_collection['hr_pearl_class'] == 'Contrafactual') & (golden_collection['hr_pearl_class'] == golden_collection['pearl_class'])].shape[0]
    divisor = golden_collection[(golden_collection['hr_pearl_class'] == 'Contrafactual')].shape[0]
    
    recall['pearl_class_Contra'] = round(dividend/divisor, 2)
    
    return pd.Series(recall)

def calculate_f1_score(precision: pd.Series, recall: pd.Series, source='all') -> pd.Series:
    '''
    This function generates F1-Score for the Golden Collection. If no source is provided, will use all
    data.
    
    Parameters:
    : precision - Series with all Precision metrics.
    : recall - Series with all Recall metrics.
    : source - Data source used.
    
    Return: Series with metrics for all classes.
    '''
    
    f1_score = {
        "source": source,
        "metric": 'F1 Score',
        "causality": 0.0,
        "non_causality": 0.0,
        "action_class_SCause": 0.0,
        "action_class_SEffect": 0.0,
        "action_class_SRelation": 0.0,
        "action_class_SSteps": 0.0,
        "action_class_SRecomm": 0.0,
        "pearl_class_Assocc": 0.0,
        "pearl_class_Interv": 0.0,
        "pearl_class_Contra": 0.0,
    }
    
    for key in f1_score.keys():
        dividend = 2*precision[key]*recall[key]
        divisor = precision[key]+recall[key]
        
        recall[key] = round(dividend/divisor, 2)
    
    return pd.Series(f1_score)