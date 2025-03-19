import base64


def get_headers_speech_to_text():
    return {
        "Authorization": "Bearer ya29.a0AeXRPp7oa5lPtbEYkJSjmveGN-VCMdxHTk2PomV7sLV6sFj2THWyF_0_QF6QT1dSu7VWfNmDlA-FSmM5avvEjENMSyRB92woxBbDhnMgrsrrG-iRhQ2BKJFRudxOr8QRVWISvKYPOF4NXCCPneZ-hgV6zL909jlsnUfxIJxCNUHnPV3rSNj5umhXj8DSZWT0e7PS50-7hKROQ7UA76ZN5ZCdsUSog_6iBpkn7ovtuyztR4BM5thNehckswyrXthwLHTcCExDyu15oQ6NYbCtsiP1sdbQOE7nM0y-vH2QRym9xhtGnzyLmF3BTnAkoXpMpT81Ybms8rt-cuJ1vHh_1ysLZCk8OIi0oGhNYPJ5bpC1S38xhB6OkoIWSVIXh_uDkE0ZKaHGSL7cIiqOS_2cVdvKaUla1rmc1BGQAAaCgYKAcYSARISFQHGX2MiYwiilKaZvg1M6wlO11hM2Q0429",
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
