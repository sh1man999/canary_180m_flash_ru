from nemo.collections.asr.models import EncDecMultiTaskModel


def main():
    canary_model = EncDecMultiTaskModel.restore_from(
        './canary_results/canary-180m-flash-finetune/checkpoints/canary-180m-flash-finetune.nemo')
    output = canary_model.transcribe(['test.wav'],
                                     source_lang='ru',
                                     target_lang='ru',
                                     pnc='yes')
    print(output[0].text)


if __name__ == "__main__":
    main()
