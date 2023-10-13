import google.generativeai as palm
from CONSTANTS import Api_key
import pandas as pd

# list of strings
lst = ['Cat', 'Dog', 'Person', 'Mobile',
       'Remote', 'Bottle', 'Orange']

l2 = [1, 2, 4, 5, 6, 8, 7]
s = {'objects': lst, "distance": l2}

# Calling DataFrame constructor on list
df = pd.DataFrame(data=s, index=[x for x in range(len(lst))])

grouped_mean_df = df.groupby(['objects']).mean() / 12

df_string = grouped_mean_df.to_string()



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
    return response.result

if __name__ == "__main__":
    print(createSceneFromEnv(df_string))
