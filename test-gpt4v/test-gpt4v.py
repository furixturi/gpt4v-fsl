# Official Doc Quickstarts > GPT-4 Turbo with Vision
# https://learn.microsoft.com/en-us/azure/ai-services/openai/gpt-v-quickstart?tabs=image%2Cbash&pivots=programming-language-python

# Using Australia East region
# As of 2024/3/29, GPT-4 vision preview is only available in australiaeast, japaneast, swedencentral, switzerlandnorth, westus 
# Ref: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-preview-model-availability)

from dotenv import load_dotenv
load_dotenv()

import base64, json, os, requests
from openai import AzureOpenAI

api_key = os.getenv("AZURE_OPENAI_API_KEY_AU")
api_version=os.getenv("AZURE_OPENAI_API_VERSION_PREVIEW")
model_endpoint = f'{os.getenv("AZURE_OPENAI_ENDPOINT_AU").strip()}openai/deployments/{os.getenv("AZURE_OPENAI_MODEL_GPT4_V").strip()}/chat/completions?api-version={os.getenv("AZURE_OPENAI_API_VERSION_PREVIEW").strip()}'
print(model_endpoint)
# model_endpoint = "https://aoai-quickstart-alex-au.openai.azure.com/openai/deployments/gpt-4-vision-10k/chat/completions?api-version=2024-02-15-preview"


# encode local images in base64 to pass to gpt-4v
# https://platform.openai.com/docs/guides/vision
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

headers = {
    "Content-Type": "application/json",
    "api-key": api_key,
}

local_image = './images/parking_lot.png'
encoded_image = base64.b64encode(open(local_image, 'rb').read()).decode('ascii')

payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful multilingual assistant trained to interpret images and can make responsible assumptions about people and places."},
        {"role": "user", "content": [
                {"type": "text", "text": "What's in this picture:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
            ]
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 2000
}

try:
    response = requests.post(model_endpoint, headers=headers, json=payload)
    response.raise_for_status()
except requests.RequestException as e:
    raise SystemExit(f'Failed to make the request. Error: {e}')

print(json.dumps(response.json(), indent=4))
print(response.json()['choices'][0]['message']['content'])

######## Output 2024/03/29 ########
# The picture contains three different boards displaying parking rates in Japan. Unfortunately, certain information on the boards appears to be obscured, likely to protect privacy or sensitive information. Each board has designated day and night hours with corresponding parking rates, indicating that the fees vary depending on the time of day. The text is in Japanese, with prices listed in Japanese yen (¥).

# - The first board on the left advertises rates such as ¥2,000 until 22:00-7:00, a ¥500 fee for additional time, and a mention of a 20-minute rate for ¥300 among other details.
# - The middle board includes daytime rates from 8:00-20:00 at ¥3,000, night-time rates from 20:00-8:00 at ¥400, and shorter intervals such as 20 minutes for ¥400 and 60 minutes for ¥100.
# - The board on the right also shows day (8:00-20:00) and night (20:00-8:00) rates with varying prices such as ¥4,620 and ¥550 for certain durations, along with a 24-hour rate.

# These signs are commonly found in urban parking areas to inform drivers of the parking fees they can expect to incur.
######## End of output ##########