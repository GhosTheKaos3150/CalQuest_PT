import requests
from tqdm import tqdm
import pycld2 as cld2

def get_first_user_content(conversation):
    '''
    This functions gets the first user LLM message, namely the "question" of the conversation.
    This is meant to be used on LLMs conversations only.
    
    Parameters:
    : conversation - List of dialogs with LLM.
    
    Return: First user LLM message ("question") or None, if none is found.
    '''
    
    for message in conversation:
        if message['role'] == 'user':
            return message['content']
    return None

def get_first_bot_content(conversation):
    '''
    This functions gets the first LLM response, namely the "answer" of the conversation.
    This is meant to be used on LLMs conversations only.
    
    Parameters:
    : conversation - List of dialogs with LLM.
    
    Return: First LLM response ("answer") or None, if none is found.
    '''
    
    for message in conversation:
        if message['role'] == 'assistant':
            return message['content']
    return None

def is_portuguese(text: str):
    '''
    Detects if the text is in Portuguese or not.
    
    NOTE: It is possible that the text detection gives false-positives and points other language content
    as being Portuguese. It is necessary to manualy analyse the data after this step.
    
    Parameters:
    : text - Text to be analysed.
    
    Return: True if text is in Portuguese or False if it is not.
    '''
    
    try:
        _, _, details = cld2.detect(text)
        return details[0][1] == 'pt'
    except:
        return False
    
def download_file(url: str, output_path: str):
    '''
    Download files from ShareGPT.
    
    Parameters:
    : url - URL used to download file.
    : output_path - Path used to save the file.
    
    Return:
    '''
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(output_path, 'wb') as file:
        for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB', unit_scale=True):
            file.write(data)