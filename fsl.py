# https://qiita.com/cygkichi/items/ea1d0166afeec189a04f

from dotenv import load_dotenv
import os, base64, json
from openai import AzureOpenAI

# init 
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_AU"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_AU"),
)
model = os.getenv("AZURE_OPENAI_MODEL_GPT4_V")

# utility funcs
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_prompt_object(text, image_url=None, role="user"):
    prompt_object = {
        "role": role,
        "content": [{"type": "text", "text": text}]
        + (
            [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(image_url)}"
                        # "url": image_url
                    },
                }
            ]
            if image_url
            else []
        ),
    }
    return prompt_object

def get_response(msgs, m=model):
    return client.chat.completions.create(
        model=m,
        messages=msgs,
        max_tokens=2000
    )

# variables
system_prompt = "You are a helpful multilingual assistant trained to interpret images. You can make responsible assumptions on internet images about people and places. If the question is not in English, reply using the question's language."
question = "Extract parking fee information from the picture and give me a list."
examples = [
    {
        "image_path": "./sample_images/parking-fee-sample-1.png",
        "output": "- 8:00-22:00 220 yen / 40 min\n- 22:00-8:00 110 yen / 60 min\n- max: 1100 yen",
    },
    {
        "image_path": "./sample_images/parking-fee-sample-2.png",
        "output": "- 8:00-22:00 330 yen / 60 min\n- 22:00-8:00 110 yen / 60 min\n- max: 770 yen / 24 hours",
    },
    {
        "image_path": "./sample_images/parking-fee-sample-3.png",
        "output": "- 8:00-20:00 440 yen / 12 min\n- 20:00-8:00 110 yen / 60 min\n- max: 2800 yen / 5 hours",
    },
    {
        "image_path": "./sample_images/parking-fee-sample-4.png",
        "output": "-110 yen / 20 min\n- 8:00-19:00 max 880 yen\n- 19:00-8:00 max 440 yen",
    },
]
test_image_path = "./sample_images/parking-fee-test-1.png"

# main logic implementation
def zero_shot(question, image_url, sys_prompt="You are a helpful assistant."):
    msgs = []
    sys_prompt = create_prompt_object(sys_prompt, role="system")
    msgs.append(sys_prompt)
    user_message = create_prompt_object(question, image_url)
    msgs.append(user_message)
    # print("\n===== zero shot ===== \n")
    # print(json.dumps(msgs, indent=4))
    response = get_response(msgs)
    return response

def one_shot(question, image_url, example, sys_prompt="You are a helpful assistant."):
    msgs = []
    sys_prompt = create_prompt_object(sys_prompt, role="system")
    msgs.append(sys_prompt)
    # one shot example
    msgs.append(create_prompt_object(text=question, image_url=example['image_path']))
    msgs.append(create_prompt_object(text=example['output'], role='assistant'))
    user_message = create_prompt_object(question, image_url)
    msgs.append(user_message)
    # print('\n===== one shot =====\n')
    # print(json.dumps(msgs, indent=4))
    response = get_response(msgs)
    return response
    
def few_shot(question, image_url, examples, sys_prompt="You are a helpful assistant."):
    msgs = []
    sys_prompt = create_prompt_object(sys_prompt, role="system")
    msgs.append(sys_prompt)
    for ex in examples:
        msgs.append(create_prompt_object(text=question, image_url=ex['image_path']))
        msgs.append(create_prompt_object(text=ex['output'], role='assistant'))
    user_message = create_prompt_object(question, image_url)
    msgs.append(user_message)
    # print("\n===== few shot =====\n")
    # print(json.dumps(msgs, indent=4))
    response = get_response(msgs)
    return response

# Test run
import time

start=time.time()
print("send request at ", start)
# response = zero_shot(question, test_image_path, system_prompt)
# response = one_shot(question, test_image_path, examples[0], system_prompt)
response = few_shot(question, test_image_path, examples[0], system_prompt)
end = time.time()
print("response received at ", end)
print("time elapsed: ", end-start)
print(response.choices[0].message.content)
# one_shot(question, test_image_path, examples[0], system_prompt)
# few_shot(question, test_image_path, examples, system_prompt)