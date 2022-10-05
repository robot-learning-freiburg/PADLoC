from collections import namedtuple
from pathlib import Path

from evaluation_comparison.plot_path_from_pairs import main as plot_pairs
from evaluation_comparison.plot_styles import Style

PairFile = namedtuple("pair_file", ["label", "file", "dataset", "seq", "is_distance"])
Dataset = namedtuple("dataset", ["path"])


def main():
    # dataset_path = Path("/data/arceyd/kitti/")
    home_dir = Path("/work/dlclarge2/arceyd-padloc/")
    save_dir = home_dir / "res" / "img" / "lcd"

    pair_dir = home_dir / "res" / "final_models"

    pair_file_ext = ".npz"
    fig_ext = ".png"

    pair_files = [
        # Kitti
        PairFile(label="overlap_kitt_seq_08",
                 file=pair_dir / "baselines" / "overlap" / "kitt" / "lcd" / "overlap_pairs_08",
                 dataset="kitti", seq="08", is_distance=False),
        PairFile(label="lcdnet_210916000234_lastiter_kitt_seq_08",
                 file=pair_dir / "lcdnet_210916000234" / "lastiter" / "kitt" / "lcd" /
                 "eval_lcd_lcdnet_210916000234_lastiter_kitt_seq_08",
                 dataset="kitti", seq="08", is_distance=True),
        PairFile(label="padloc_220527191054_lastiter_kitt_seq_08",
                 file=pair_dir / "padloc_220527191054" / "lastiter" / "kitt" / "lcd" /
                 "eval_lcd_padloc_220527191054_lastiter_kitt_seq_08",
                 dataset="kitti", seq="08", is_distance=True)
        # PairFile("dcp_kitti_seq00", "kitti", "00"),
        # PairFile("dcp_kitti_seq08", "kitti", "08"),
        # PairFile("lcdnet_kitti_seq00", "kitti", "00"),
        # PairFile("lcdnet_kitti_seq08", "kitti", "08"),
        # PairFile("padloc_kitti_seq00", "kitti", "00"),
        # PairFile("padloc_kitti_seq08", "kitti", "08"),
        # PairFile("tf_kitti_seq00", "kitti", "00"),
        # PairFile("tf_kitti_seq08", "kitti", "08"),
        # # Kitti360
        # PairFile("dcp_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
        # PairFile("lcdnet_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
        # PairFile("padloc_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
        # PairFile("tf_kitti360_seq2013_05_28_drive_0002_sync", "kitti360", "2013_05_28_drive_0002_sync"),
    ]

    datasets = {
        "kitti": Path("/home/arceyd/MT/dat/kitti/dataset"),  # home_dir / "dat" / "KITTI",
        "kitti360": Path("/home/arceyd/MT/dat/kitti360/")
    }

    styles = [
        # Style("bw"),
        Style("color", use_latex=False)
    ]

    for pair_file in pair_files:
        file_path = home_dir / f"{pair_file.file}{pair_file_ext}"
        save_path = save_dir / f"fig_lcd_{pair_file.label}{fig_ext}"
        plot_pairs(file_path,
                   dataset=pair_file.dataset, dataset_path=datasets[pair_file.dataset], sequence=pair_file.seq,
                   styles=styles, save_path=save_path, do_plot_legends=True, is_distance=pair_file.is_distance)


if __name__ == "__main__":
    main()
