from nemo.collections.asr.models import EncDecMultiTaskModel


def main():
    canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-180m-flash')
    output = canary_model.transcribe(['test.wav'])
    print(output[0].text)


if __name__ == "__main__":
    main()
