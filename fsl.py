# https://qiita.com/cygkichi/items/ea1d0166afeec189a04f

from dotenv import load_dotenv

load_dotenv()

import os, base64, json
from openai import AzureOpenAI


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_AU"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_AU"),
)
model = os.getenv("AZURE_OPENAI_MODEL_GPT4_V")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


sys_prompt = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You are a helpful multilingual assistant trained to interpret images and make responsible assumptions on internet images about people and places. If the question is not in English, reply using the question's language.",
        }
    ],
}

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


few_shot_messages = []
for ex in examples:
    few_shot_messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(ex['image_path'])}"
                    },
                },
            ],
        }
    )
    few_shot_messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": ex["output"]}]}
    )

# print(json.dumps(few_shot_messages, indent=4))

test_image_path = "./sample_images/parking-fee-test-1.png"
messages = (
    [sys_prompt]
    + few_shot_messages
    + [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(test_image_path)}"
                    },
                },
            ],
        }
    ]
)

# response = client.chat.completions.create(
#     model=model, messages=messages, max_tokens=800
# )

# print(response.choices[0].message.content)


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

system_prompt = "You are a helpful multilingual assistant trained to interpret images. You can make responsible assumptions on internet images about people and places. If the question is not in English, reply using the question's language."


def get_response(msgs, m=model):
    return client.chat.completions.create(
        model=m,
        messages=msgs,
        max_tokens=2000
    )
    
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
    
    


# response = zero_shot(question, test_image_path, system_prompt)
# print(response.choices[0].message.content)

zero_shot(question, test_image_path, system_prompt)
one_shot(question, test_image_path, examples[0], system_prompt)
few_shot(question, test_image_path, examples, system_prompt)