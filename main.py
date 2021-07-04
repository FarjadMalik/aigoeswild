import config
import trainer
import evaluator


def main(mode=None):
    if mode == 'train':
        trainer.train()
    elif mode == 'evaluate':
        evaluator.evaluate()
    else:
        raise Exception(f'config.mode should be either "train" or "evaluate". Not {mode}')
    pass


if __name__ == '__main__':
    print(f'---Major Tom to Ground Control---')
    assert (config.mode is not None)
    main(config.mode)
    print(f'------')
