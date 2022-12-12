from argparse import ArgumentParser
from pathlib import Path
from models.get_models import load_model

if __name__ == "__main__":

    root_path = Path("/home/arceyd/MT/cp/3D/")

    cp_paths = {
        "LCDNet": root_path / "16-09-2021_00-02-34/checkpoint_last_iter.tar",
        "DCP": root_path / "04-04-2022_18-34-14/checkpoint_last_iter.tar",
        "PADLOC": root_path / "27-05-2022_19-10-54/checkpoint_last_iter.tar",
    }

    def count_params(cp_path):
        model, _ = load_model(cp_path)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return pytorch_total_params

    for k, cp in cp_paths.items():
        n_params = count_params(cp)
        print(f"Model {k} has trainable {n_params} parameters.")

    pass
