import os
import torch


def prepare_baseline(checkpoint_path, path_to_save):
    from models.resnet50 import Resnet50

    model = Resnet50(embed_size=128)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    traced_model = torch.jit.trace(model, torch.rand((1, 3, 224, 224)))
    torch.jit.save(traced_model, path_to_save)


def prepare_dc(checkpoint_path, path_to_save):
    import divide_and_conquer.lib as lib
    config = {
        'sz_embedding': 128,
        'random_seed': 0,
        'nb_clusters': 8
    }
    model = lib.model.make(config)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    traced_model = torch.jit.trace(model, torch.rand((1, 3, 224, 224)))
    torch.jit.save(traced_model, path_to_save)


def prepare_xbm(checkpoint_path, path_to_save):
    from ret_benchmark.modeling import build_model
    from ret_benchmark.config import cfg
    from ret_benchmark.utils.model_serialization import load_state_dict
    model = build_model(cfg)
    state_dict = torch.load(checkpoint_path)
    load_state_dict(model, state_dict['model'])
    model.eval()
    traced_model = torch.jit.trace(model, torch.rand((1, 3, 224, 224)))
    torch.jit.save(traced_model, path_to_save)


if __name__ == '__main__':
    # prepare_baseline(
    #     r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\checkpoints\baseline\FIN-59__epoch-10.pt',
    #     r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\demo\models\FIN-59_traced.pt')

    prepare_dc(
        r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\checkpoints\divide_and_conquer\flowers-K-8-M-2-exp-0.pt',
        r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\demo\models\FIN-72_traced_with_norm.pt'
    )

    # prepare_xbm(
    #     r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\checkpoints\xbm\model_004914.pth',
    #     r'C:\Users\khanz\PycharmProjects\inno_ds\final_project\data\demo\models\FIN-71_traced_last.pt'
    # )
