import pandas as pd

def generate_inference_metrics(df: pd.DataFrame):
    # Total
    all_precision = calculate_precision(df)
    all_recall = calculate_recall(df)
    all_f1_score = calculate_f1_score(all_precision, all_recall)
    
    df_metrics = pd.concat([all_precision, all_recall, all_f1_score])
    
    # By Source
    for source in ['reddit', 'wildchat', 'sharegpt']:
        source_precision = calculate_precision(df[df['source'] == source], source)
        source_recall = calculate_recall(df[df['source'] == source], source)
        source_f1_score = calculate_f1_score(all_precision, all_recall, source)
        
        df_metrics = pd.concat([df_metrics, source_precision, source_recall, source_f1_score])
    
    df_metrics.sort_values(by='metric')
    df_metrics.to_excel('./metrics_gen/golden_collect_metrics.xlsx')

    return df_metrics

def calculate_precision(df: pd.DataFrame, source='all') -> pd.Series:
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
    dividend = df[(df['hr_is_causal'] == True) & (df['hr_is_causal'] == df['is_causal'])].shape[0]
    divisor = df[(df['is_causal'] == True)].shape[0]
    
    precisions['causality'] = round(dividend/divisor, 2)
    
    # Non Causality
    dividend = df[(df['hr_is_causal'] == False) & (df['hr_is_causal'] == df['is_causal'])].shape[0]
    divisor = df[(df['is_causal'] == False)].shape[0]
    
    precisions['non_causality'] = round(dividend/divisor, 2)

    # Action Class (Seek-Cause)
    dividend = df[(df['hr_action_class'] == 'Busca-Causa') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['action_class'] == 'Busca-Causa')].shape[0]
    
    precisions['action_class_SCause'] = round(dividend/divisor, 2)

    # Action Class (Seek-Effect)
    dividend = df[(df['hr_action_class'] == 'Busca-Efeito') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['action_class'] == 'Busca-Efeito')].shape[0]

    precisions['action_class_SEffect'] = round(dividend/divisor, 2)

    # Action Class (Seek-Relation)
    dividend = df[(df['hr_action_class'] == 'Busca-Relação') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['action_class'] == 'Busca-Relação')].shape[0]
    
    precisions['action_class_SRelation'] = round(dividend/divisor, 2)

    # Action Class (Seek-Steps)
    dividend = df[(df['hr_action_class'] == 'Busca-Passos') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['action_class'] == 'Busca-Passos')].shape[0]
    
    precisions['action_class_SSteps'] = round(dividend/divisor, 2)

    # Action Class (Seek-Recomm)
    dividend = df[(df['hr_action_class'] == 'Busca-Recomendação') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['action_class'] == 'Busca-Recomendação')].shape[0]
    
    precisions['action_class_SRecomm'] = round(dividend/divisor, 2)

    # Pearl Class (Assocciative)
    dividend = df[(df['hr_pearl_class'] == 'Associacional') & (df['hr_pearl_class'] == df['pearl_class'])].shape[0]
    divisor = df[(df['pearl_class'] == 'Associacional')].shape[0]
    
    precisions['pearl_class_Assocc'] = round(dividend/divisor, 2)

    # Pearl Class (Intervencionist)
    dividend = df[(df['hr_pearl_class'] == 'Intervencional') & (df['hr_pearl_class'] == df['pearl_class'])].shape[0]
    divisor = df[(df['pearl_class'] == 'Intervencional')].shape[0]
    
    precisions['pearl_class_Interv'] = round(dividend/divisor, 2)

    # Pearl Class (Contrafactual)
    dividend = df[(df['hr_pearl_class'] == 'Contrafactual') & (df['hr_pearl_class'] == df['pearl_class'])].shape[0]
    divisor = df[(df['pearl_class'] == 'Contrafactual')].shape[0]
    
    precisions['pearl_class_Contra'] = round(dividend/divisor, 2)
    
    return pd.Series(precisions)

def calculate_recall(df: pd.DataFrame, source='all') -> pd.Series:
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
    dividend = df[(df['hr_is_causal'] == True) & (df['hr_is_causal'] == df['is_causal'])].shape[0]
    divisor = df[(df['hr_is_causal'] == True)].shape[0]
    
    recall['causality'] = round(dividend/divisor, 2)
    
    # Non Causality
    dividend = df[(df['hr_is_causal'] == False) & (df['hr_is_causal'] == df['is_causal'])].shape[0]
    divisor = df[(df['hr_is_causal'] == False)].shape[0]
    
    recall['non_causality'] = round(dividend/divisor, 2)

    # Action Class (Seek-Cause)
    dividend = df[(df['hr_action_class'] == 'Busca-Causa') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['hr_action_class'] == 'Busca-Causa')].shape[0]
    
    recall['action_class_SCause'] = round(dividend/divisor, 2)

    # Action Class (Seek-Effect)
    dividend = df[(df['hr_action_class'] == 'Busca-Efeito') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['hr_action_class'] == 'Busca-Efeito')].shape[0]
    
    recall['action_class_SEffect'] = round(dividend/divisor, 2)

    # Action Class (Seek-Relation)
    dividend = df[(df['hr_action_class'] == 'Busca-Relação') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['hr_action_class'] == 'Busca-Relação')].shape[0]
    
    recall['action_class_SRelation'] = round(dividend/divisor, 2)

    # Action Class (Seek-Steps)
    dividend = df[(df['hr_action_class'] == 'Busca-Passos') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['hr_action_class'] == 'Busca-Passos')].shape[0]
    
    recall['action_class_SSteps'] = round(dividend/divisor, 2)

    # Action Class (Seek-Recomm)
    dividend = df[(df['hr_action_class'] == 'Busca-Recomendação') & (df['hr_action_class'] == df['action_class'])].shape[0]
    divisor = df[(df['hr_action_class'] == 'Busca-Recomendação')].shape[0]
    
    recall['action_class_SRecomm'] = round(dividend/divisor, 2)

    # Pearl Class (Assocciative)
    dividend = df[(df['hr_pearl_class'] == 'Associacional') & (df['hr_pearl_class'] == df['pearl_class'])].shape[0]
    divisor = df[(df['hr_pearl_class'] == 'Associacional')].shape[0]
    
    recall['pearl_class_Assocc'] = round(dividend/divisor, 2)

    # Pearl Class (Intervencionist)
    dividend = df[(df['hr_pearl_class'] == 'Intervencional') & (df['hr_pearl_class'] == df['pearl_class'])].shape[0]
    divisor = df[(df['hr_pearl_class'] == 'Intervencional')].shape[0]
    
    recall['pearl_class_Interv'] = round(dividend/divisor, 2)

    # Pearl Class (Contrafactual)
    dividend = df[(df['hr_pearl_class'] == 'Contrafactual') & (df['hr_pearl_class'] == df['pearl_class'])].shape[0]
    divisor = df[(df['hr_pearl_class'] == 'Contrafactual')].shape[0]
    
    recall['pearl_class_Contra'] = round(dividend/divisor, 2)
    
    return pd.Series(recall)

def calculate_f1_score(precision: dict, recall: dict, source='all') -> pd.Series:
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