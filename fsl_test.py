from fsl import FewShotLearning
import asyncio

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
