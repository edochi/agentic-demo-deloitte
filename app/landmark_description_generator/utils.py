import base64


def get_headers_speech_to_text():
    return {
        "Authorization": "Bearer ya29.a0AeXRPp6S9q6RStfooTtAfy37Z-LYJC718sa63qRE_JJ_pUlflH22RXWwUQ5oAfULy9KYom_DCzORGVhk23bjhcnq2UEVS4lZ3eUwmw2bKkD6yv3TPeidZc-rNIG4WGlxaan0RJ1Y6g2O8WuxwW5JYS56F00A0iRb4Hw1MoJ_-fcnAS1nzvX6pbLJt97sO6cnerVxNBU2VAP5SE1DliovG1TyNkCFmjyFMtEb-KIpnOHGq_fadkCpvjKxqI27m9yhX0hDEaEqverWjemTbqg8QKfMbSZdc6VvUIWnbyBeKk6132Vxm6iTeBuqIB9vgmsH3iDE-ZA1pSl8aw5U6pQzO5x8GArkW4AduOnKjJ6-Y2d9yJ7qKv-MnX9pq-4gxjr6abLPdbELRaScBahIdjZhMF-UlN1_ATK-WwaCgYKAU4SARISFQHGX2MisE8-KkGr7jnVNRCMpGw6fA0425",
        "x-goog-user-project": "qwiklabs-gcp-02-44d130f8f4a0",
        "Content-Type": "application/json; charset=utf-8",
    }


def get_body_speech_to_text(input_text: str):
    return {
        "input": {"text": input_text},
        "voice": {
            "languageCode": "en-gb",
            "name": "en-GB-Standard-A",
            "ssmlGender": "FEMALE",
        },
        "audioConfig": {"audioEncoding": "MP3"},
    }


def get_decoded_body_from_respopnse(response):
    return base64.b64decode(response["audioContent"])
