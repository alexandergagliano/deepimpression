import cerebras_pytorch as cstorch

import sys
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", help="model directory to use", type=str, required=True)
    parser.add_argument("--config", help="The config file to use")
    parser.add_argument("--module_dirs", help="Additional module paths", nargs='+', default=[])
    args = parser.parse_args()

    for module_dir in args.module_dirs:
        sys.path.append(module_dir)

    from deepimpression.utils.util import load_model, deserialize

    # Load parameters
    model, params = load_model(
        config=args.config,
        modeldir=args.modeldir)
    params['train_config']['modeldir'] = args.modeldir

    compiled_model = cstorch.compile(model, backend="CSX")

    train_dl_func = deserialize(params['train_input']['func'])
    train_dl = cstorch.utils.data.DataLoader(train_dl_func, **params)

    test_dl_cls = deserialize(params['eval_input']['func'])
    test_dl = cstorch.utils.data.DataLoader(test_dl_cls, **params)

    train_func = deserialize(params['train_config']['func'])

    train_func(params, model, train_dl, test_dl)


if __name__ == "__main__":
    main()
