# %%
print("This is cell 1")
import pandas as pd
import requests
def fetch_customer_feedback():
    """
    Fetches raw customer feedback/comments from a public REST API.
    Includes proper headers and error handling for production resilience.
    """
    url = "https://jsonplaceholder.typicode.com/comments"
    header ={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0"}

    try: 

        response = requests.get(url, headers=header,timeout = 10)
        response.raise_for_status()
        raw_data = response.json()
        df = pd.DataFrame(raw_data)
        # Rename columns to match a real customer feedback business use-case
        df = df.rename(colums={
            "postId" : "product_id",
            'id':'feedback_id',
            'name':'customer_name',
            'email' :'customer_email',
            'body':'Review_text'

        })
        return df[['product_id','feedback_id','customer_name','customer_email','Review_text']
                  ]
    except requests.exceptions.RequestException as e:
        print(f'error fetching date:{e}')
        return pd.DataFrame


# %%
print("This is cell 3")
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "llama-3.3-70b-versatile", # Great flagship model on Groq
    "messages": [{"role": "user", "content": "Hello, respond with one word."}]
}

try:
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print("✅ Groq API Key is working perfectly!")
        # Print the exact rate limit details from the response headers
        print(f"Requests Remaining Today: {response.headers.get('x-ratelimit-remaining-requests')}")
        print(f"Tokens Remaining This Minute: {response.headers.get('x-ratelimit-remaining-tokens')}")
    elif response.status_code == 429:
        print("❌ Groq limit exceeded (Rate Limited / Expired for this minute).")
        print(f"Retry after: {response.headers.get('retry-after')} seconds")
    else:
        print(f"❌ Error: Status Code {response.status_code}, Response: {response.text}")

except Exception as e:
    print(f"Connection failed: {e}")
# %%
print("This is cell 4")

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import requests
import pandas as pd

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"))


def fetch_customer_feedback():
    """Fetches raw customer feedback/comments from a public REST API"""
    url ="https://jsonplaceholder.typicode.com/comments"
    try:
        response = requests.get(url,timeout=10)
        response.raise_for_status()
        df =pd.DataFrame(response.json()[:10])
        return df.rename(columns={'id': 'feedback_id', 'name': 'review_title', 'email': 'customer_email', 'body': 'review_text'})
    except Exception as e:
        return pd.DataFrame()
def analyze_review_text(review_text:str) ->dict:
    try:
        prompt_instruction = (
            "Analyze the customer review text provided and  you must return your output"
            "Strictly as a single, valid json object with nothing else.do not include markdown block formatting"
            "Do NOT wrap it in markdown code blocks like ```json ... ```. "
            "The JSON keys must be exactly:\n"
            "- 'sentiment': (Value must be 'Positive', 'Neutral', or 'Negative')\n"
            "- 'severity_score': (An integer from 1 to 5)\n"
            "- 'primary_topic': (Main category, e.g., UI/UX, Bug, Support, Pricing)\n"
            "- 'summary_sentence': (A clear 1-sentence summary)\n"
        )
        completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages =[{"role":"system","content":prompt_instruction},
                   {"role": "user", "content": review_text}],
                   temperature =0.1

         )
        raw_content = completion.choices[0].message.content.strip()
    
        if "```" in raw_content:
            # Extract everything between the first { and the last }
            start_idx = raw_content.find("{")
            end_idx = raw_content.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                raw_content = raw_content[start_idx:end_idx]

        return json.loads(raw_content)
    except Exception as e:
        print(f"Groq Processing Error for text [{review_text[:20]}...]: {e}")
        return{
            "sentiment" : "Neutral",
            "severity_score":3,
            "primary_topic":"Bug",
            "summary_sentence":"Analysis failed due to response parsing issue."
        }
def process_full_feedback_pipeline():
    df = fetch_customer_feedback()
    if df.empty:
        return df
    
    analysis_results = [analyze_review_text(text) for text in df['review_text']]
    df_analysis = pd.DataFrame(analysis_results)
    return pd.concat([df, df_analysis], axis=1)




