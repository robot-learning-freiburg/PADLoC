from itertools import combinations
from pathlib import Path


SEQUENCES = [
	"2020-11-26-13-58",	 # Overcast, Snow
	"2020-12-01-13-26",	 # Overcast, Snow, Snowing
	"2020-12-04-14-00",	 # Overcast, Snow
	"2020-12-18-13-44",	 # Sun, Snow
	"2021-01-15-12-17",	 # Sun, Clouds, Snow
	"2021-01-19-15-08",	 # Clouds, Snow
	"2021-01-26-10-59",	 # Overcast, Snow, Snowing
	"2021-01-26-11-22",	 # Overcast, Snow, Snowing
	"2021-02-02-14-07",	 # Overcast, Snow
	"2021-02-09-12-55",	 # Sun, Clouds, Snow
	"2021-03-02-13-38",	 # Sun, Clouds, Snow
	"2021-03-09-14-23",	 # Sun
	"2021-03-23-12-43",	 # Overcast, Construction
	"2021-03-30-14-23",	 # Sun, Clouds, Construction
	"2021-04-08-12-44",	 # Sun
	"2021-04-13-14-49",	 # Sun, Clouds, Construction
	"2021-04-15-18-55",	 # Clouds, Construction
	"2021-04-20-14-11",	 # Clouds, Construction
	"2021-04-22-15-00",	 # Clouds, Snowing, Construction
	"2021-04-29-15-55",	 # Overcast, Rain
	"2021-05-06-13-19",	 # Sun, Clouds
	"2021-05-13-16-11",	 # Sun, Clouds
	"2021-06-03-16-00",	 # Sun, Clouds
	"2021-06-17-17-52",	 # Sun
	"2021-06-29-18-53",	 # Overcast, Rain
	"2021-06-29-20-43",	 # Sun, Clouds, Dusk
	"2021-07-20-17-33",	 # Clouds, Rain
	"2021-07-27-14-43",	 # Clouds
	"2021-08-05-13-34",	 # Sun, Clouds
	"2021-09-02-11-42",	 # Sun
	"2021-09-07-09-35",	 # Sun
	"2021-09-08-21-00",	 # Night
	"2021-09-09-15-28",	 # Sun, Clouds, Alternate, Construction
	"2021-09-14-20-00",	 # Night
	"2021-10-05-15-35",	 # Overcast
	"2021-10-15-12-35",	 # Clouds
	"2021-10-22-11-36",	 # Clouds
	"2021-10-26-12-35",	 # Overcast, Rain
	"2021-11-02-11-16",	 # Sun, Clouds
	"2021-11-06-18-55",	 # Night
	"2021-11-14-09-47",	 # Overcast
	"2021-11-16-14-10",	 # Clouds
	"2021-11-23-14-27",	 # Sun, Clouds
	"2021-11-28-09-18",  # Overcast, Snow, Snowing
]

SEQUENCE_PAIRS = list(combinations(SEQUENCES, 2))


def out_filename(seq1, seq2, model_name, path, proc, dataset="boreas", ext="pickle"):
	return path / f"{model_name}_{dataset}_{proc}_seqs_{seq1}_{seq2}.{ext}"


def yaw_stats_filename(seq1, seq2, model_name, path, dataset="boreas", ext="pickle"):
	return out_filename(seq1, seq2, model_name, path, "inference_yaw_stats", dataset, ext)


def pr_filename(seq1, seq2, model_name, path, dataset="boreas", ext="npz"):
	return out_filename(seq1, seq2, model_name, path, "inference_pr", dataset, ext)


def lcd_stats_filename(seq1, seq2, model_name, path, dataset="boreas", ext="pickle"):
	return out_filename(seq1, seq2, model_name, path, "inference_pr", dataset, ext)


def seq_pair_downloaded(seq1, seq2, path: Path):
	tmp_path1 = path / seq1
	tmp_path2 = path / seq2
	return tmp_path1.exists() and tmp_path2.exists()
