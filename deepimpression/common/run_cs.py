from mk_ic import install
install()

def main():
    from modelzoo.common.run_utils.cli_pytorch import get_params_from_args
    params = get_params_from_args()

    from modelzoo.common.pytorch.run_utils import main
    ic(params)

    from deepimpression.utils.util import deserialize

    ic(params['model']['func'], params['train_input']['func'], params['eval_input']['func'])
    #model_cls = deserialize(params['model']['class'])
    model_func = deserialize(params['model']['func'])
    train_dl = deserialize(params['train_input']['func'])
    eval_dl = deserialize(params['eval_input']['func'])

    main(
        params, model_func, train_dl, eval_dl,
    )


if __name__ == '__main__':
    main()
