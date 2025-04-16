from nemo.collections.asr.models import EncDecMultiTaskModel


def main():
    # audio max len 30 sec
    canary_model = EncDecMultiTaskModel.restore_from(
        './models/canary-180m-flash-finetune.nemo')
    output = canary_model.transcribe(['aes_l.wav'],
                                     source_lang='ru',
                                     target_lang='ru',
                                     pnc='yes')
    print(output[0].text)


if __name__ == "__main__":
    main()
