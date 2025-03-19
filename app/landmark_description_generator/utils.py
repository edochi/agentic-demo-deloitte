import base64


def get_headers_speech_to_text():
    return {
        "Authorization": "Bearer ya29.a0AeXRPp4fvBM40k8jOsCXcRM1QPrHazamHXG1zSuF5PcIFK__va8wBPo4wn1cUvtMC1_KUuGn4aphFeWguhSk6GjxlC070oDQA8d_B9PrUcAH15YFHiQH8tfO0Kj6dQKP038rVWJH87uQsjtRf9k0pH7nB95hrcelZuGklMR86wm5brX1hX2RhDhe_sT0fhzMG39trE885VgTj9ZguSqGgRPjgmAZvkslOuswgn6lk_2VNXDj6wVFr0N_C2vJn3oUrwrRz7SHoJIa0OFGFQMK2tMcnmQmUpklf74D8iAeOFe4VRSAv6TXPmr6cx3aGHQcvnNRLStynGH6Y4kcgxXbHKj6LGslIbQhDsKKU47nd2dfJEeEY_9LMJfmm5yAfiJt--oDjwaH805MzGcDwNtjaunNgmP3ctDlwhsdaCgYKAY8SARISFQHGX2MiGS7ra9cTyFGgBL4jo9LWFQ0427",
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
