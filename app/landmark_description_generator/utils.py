import base64


def get_headers_speech_to_text():
    return {
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
