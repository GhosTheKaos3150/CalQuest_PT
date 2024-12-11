import requests
from tqdm import tqdm
import pycld2 as cld2

def get_first_user_content(conversation):
    '''
    TODO Documentação
    '''
    
    for message in conversation:
        if message['role'] == 'user':
            return message['content']
    return None

def get_first_bot_content(conversation):
    '''
    TODO Documentação
    '''
    
    for message in conversation:
        if message['role'] == 'assistant':
            return message['content']
    return None

def is_portuguese(text):
    '''
    TODO Documentação
    '''
    
    try:
        _, _, details = cld2.detect(text)
        return details[0][1] == 'pt'
    except:
        return False
    
def download_file(url, output_path):
    '''
    TODO Documentação
    '''
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(output_path, 'wb') as file:
        for data in tqdm(response.iter_content(block_size), total=total_size // block_size, unit='KB', unit_scale=True):
            file.write(data)