import json
import random
import requests

#  ul = "https://hooks.slack.com/services/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" # channel: sub_b

ul = "https://hooks.slack.com/services/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"  # channel: gunmo_t1003
headers = {"Content-Type": "application/json"}

good_image_url = [
    "https://postfiles.pstatic.net/MjAxOTAxMDNfMTA0/MDAxNTQ2NDU4MjgzMzc4.Tbbk3uwOgsGucbRKYuveMaBlrSvfZB5o5CCWIiHSX2Ag.ozlEpDcXsFJgv2-62iH7NT5LmspagvKFynZCWJ0E2JYg.JPEG.dong2679/SE-ca459abc-449a-4384-a9d3-ca990c662881.jpg?type=w773",
    "https://postfiles.pstatic.net/MjAxOTAxMDNfMTQ1/MDAxNTQ2NDU4Mjg0NTgw.JjdYmgKa1j-cJsA4u_Gso3LjRrRdW7soiaPZPVVR4cYg.7WFk4ed9bJZ1kElBreIvUttn_8fSRQNiP-5L6S-RNI0g.JPEG.dong2679/SE-90154305-c63f-4f94-914c-681dca05b39f.jpg?type=w773",
    "https://images.chosun.com/resizer/0ZQzWFw22LDZF_AA2C1y4P7vnoc=/464x0/smart/cloudfront-ap-northeast-1.images.arcpublishing.com/chosun/55PRDXQN4TAWU63JGEIKMCGSPM.jpg",
    "http://haesool.com/web/product/small/20200228/9b5d7ea3154868bf3c1b6289394b2af7.png",
]

bad_image_url = [
    "http://img.dmitory.com/img/201809/5YF/T16/5YFT16dXCEQQasKua8k0yo.jpg",
    "http://img.dmitory.com/img/201809/2X8/03F/2X803FNnksK6YS6Uac8Ey8.jpg",
    "https://image.fmkorea.com/files/attach/new/20181115/486616/210097/1386568684/f889a90c82cab24e8821a9a8c5484a4c.jpg",
]


def get_format_data(text, image_url):
    data = {
        "blocks": [
            {
                "type": "section",
                "block_id": "section567",
                "text": {"type": "mrkdwn", "text": f"{text}"},
                "accessory": {
                    "type": "image",
                    "image_url": f"{image_url}",
                    "alt_text": "Joe_Boa",
                },
            }
        ]
    }

    return data


def hook_fail_strategy(strategy, error):
    text = f"@channel \n{'-'*30}\n*STRATEGY*: {strategy}\n*STATUS*: PENDING\n*ERROR_MESSAGE*: {str(error)[:100]}"
    img_url = random.choice(bad_image_url)
    data = get_format_data(text, img_url)
    res = requests.post(ul, headers=headers, data=json.dumps(data))
    print(res)


def hook_fail_ray(error):
    text = f"@channel \n{'-'*30}\n*RAY FAILED!!!* Warning!!!\n:warning::warning::warning::warning::warning::warning::warning:\n*ERROR*: {str(error)[:100]}"
    img_url = random.choice(bad_image_url)
    data = get_format_data(text, img_url)
    res = requests.post(ul, headers=headers, data=json.dumps(data))
    print(res)


def hook_simple_text(text):
    data = {
        "blocks": [
            {
                "type": "section",
                "block_id": "section567",
                "text": {"type": "mrkdwn", "text": f"{text}"},
            }
        ]
    }

    res = requests.post(ul, headers=headers, data=json.dumps(data))
    print(res)


if __name__ == "__main__":

    hook_simple_text("많이 놀랐죠?")
    pass
