import google.generativeai as palm
from CONSTANTS import Api_key
import pandas as pd


def remove_special_characters(input_string):
    special_characters_pattern = r'[^a-zA-Z0-9.\s]'
    cleaned_string = re.sub(special_characters_pattern, '', input_string)
    return cleaned_string




def createSceneFromEnv(data):
    palm.configure(api_key=Api_key)

    defaults = {
        'model': 'models/text-bison-001',
        'temperature': 0.7,
        'candidate_count': 1,
        'top_k': 40,
        'top_p': 0.95,
        'max_output_tokens': 1024,
        'stop_sequences': []
    }
    prompt1 = f"""Give environment description for blind people using an object recognition software highlighting only the distances measured(in feet) from user for entities mentioned in the list - {data} """

    response = palm.generate_text(**defaults , prompt=prompt1)
    response = remove_special_characters(response)
    return response.result


    # for sentence in cleaned_res.split('\n'):
    #      subprocess.run(['espeak', sentence])
