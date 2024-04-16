from dotenv import load_dotenv

load_dotenv()

import os, base64, json, logging, time, asyncio
# from openai import AzureOpenAI
from openai import AsyncAzureOpenAI


class FewShotLearning:
    def __init__(
        self,
        configs={},
        examples=[],
        system_prompt=None,
        question="",
        max_tokens=2000,
        test_image_url=None,
    ):
        self.client = AsyncAzureOpenAI(
            api_key=(
                configs["api_key"]
                if "api_key" in configs
                else os.getenv("AZURE_OPENAI_API_KEY_AU")
            ),
            api_version=(
                configs["api_version"]
                if "api_version" in configs
                else os.getenv("AZURE_OPENAI_API_VERSION")
            ),
            azure_endpoint=(
                configs["endpoint"]
                if "endpoint" in configs
                else os.getenv("AZURE_OPENAI_ENDPOINT_AU")
            ),
        )
        self.model = (
            configs["model"]
            if "model" in configs
            else os.getenv("AZURE_OPENAI_MODEL_GPT4_V")
        )
        self.examples = examples
        self.system_prompt = (
            system_prompt
            or "You are a helpful multilingual assistant trained to interpret images. You can make responsible assumptions on internet images about people and places. If the question is not in English, reply using the question's language."
        )
        self.question = question
        self.test_image_url = test_image_url
        self.max_tokens = max_tokens

        # log to console
        self.logger = logging.getLogger("FewShotLearning")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    # utility funcs
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_prompt_object(self, text, image_url=None, role="user"):
        prompt_object = {
            "role": role,
            "content": [{"type": "text", "text": text}]
            + (
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(image_url)}"
                            # "url": image_url
                        },
                    }
                ]
                if image_url
                else []
            ),
        }

        return prompt_object

    async def get_response(self, msgs):
        return await self.client.chat.completions.create(
            model=self.model, messages=msgs, max_tokens=self.max_tokens
        )

    # main logic implementation
    async def zero_shot(self):
        msgs = []
        sys_prompt = self.create_prompt_object(self.system_prompt, role="system")
        msgs.append(sys_prompt)
        user_message = self.create_prompt_object(self.question, self.test_image_url)
        msgs.append(user_message)

        # log request
        sent_time = time.time()
        self.logger.info("Zero shot request sent.")
        self.logger.debug("Zero shot request msgs:")
        self.logger.debug(json.dumps(msgs, indent=4))

        # send request to model endpoint
        response = await self.get_response(msgs)

        # log response
        received_time = time.time()
        self.logger.info(
            f"Zero shot response received. Time elapsed: {received_time-sent_time} seconds."
        )
        self.logger.debug("Zero shot response:")
        self.logger.debug(json.dumps(str(response), indent=4))


        return response

    async def one_shot(self, example=None):
        msgs = []
        sys_prompt = self.create_prompt_object(self.system_prompt, role="system")
        msgs.append(sys_prompt)
        # one shot example
        if not example:
            example = self.examples[0]
        msgs.append(
            self.create_prompt_object(
                text=self.question, image_url=example["image_path"]
            )
        )
        msgs.append(self.create_prompt_object(text=example["output"], role="assistant"))
        user_message = self.create_prompt_object(
            text=self.question, image_url=self.test_image_url
        )
        msgs.append(user_message)

        # log request
        sent_time = time.time()
        self.logger.info("One shot request sent.")
        self.logger.debug("One shot request msgs:")
        self.logger.debug(json.dumps(msgs, indent=4))

        # send request to model endpoint
        response = await self.get_response(msgs)

        # log response
        received_time = time.time()
        self.logger.info(
            f"One shot response received. Time elapsed: {received_time-sent_time} seconds."
        )
        self.logger.debug("One shot response:")
        self.logger.debug(response)

        return response

    async def few_shot(self):
        msgs = []
        sys_prompt = self.create_prompt_object(self.system_prompt, role="system")
        msgs.append(sys_prompt)
        # few shot examples
        for ex in self.examples:
            msgs.append(
                self.create_prompt_object(
                    text=self.question, image_url=ex["image_path"]
                )
            )
            msgs.append(self.create_prompt_object(text=ex["output"], role="assistant"))
        user_message = self.create_prompt_object(
            self.question, image_url=self.test_image_url
        )
        msgs.append(user_message)

        # log request
        sent_time = time.time()
        self.logger.info("Few shot request sent.")
        self.logger.debug("Few shot request msgs:")
        self.logger.debug(json.dumps(msgs, indent=4))

        # send request to model endpoint
        response = await self.get_response(msgs)

        # log response
        received_time = time.time()
        self.logger.info(
            f"Few shot response received. Time elapsed: {received_time-sent_time} seconds."
        )
        self.logger.debug("Few shot response:")
        self.logger.debug(response)
        return response
    
    async def run_all_shots(self, example=None):
        task_0 = asyncio.create_task(self.zero_shot())
        task_1 = asyncio.create_task(self.one_shot(example=example))
        task_few = asyncio.create_task(self.few_shot())
        responses = await asyncio.gather(task_0, task_1, task_few)
        return responses

# Test it
if __name__ == "__main__":
    # variables
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

    fsl = FewShotLearning(
        examples=examples, question=question, test_image_url=test_image_path
    )
    
    ## Sequential
    # response = asyncio.run(fsl.zero_shot())
    # print(response.choices[0].message.content)
    # response_1 = asyncio.run(fsl.one_shot())
    # print(response_1.choices[0].message.content)
    # response_2 = asyncio.run(fsl.few_shot())
    # print(response_2.choices[0].message.content)
    
    # Concurrent
    responses = asyncio.run(fsl.run_all_shots())
    for idx, res in enumerate(responses):
        print(f'==== {idx} ====')
        print(res.choices[0].message.content)
