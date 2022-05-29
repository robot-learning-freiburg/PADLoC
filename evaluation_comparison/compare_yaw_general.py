from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import pickle
from tabulate import tabulate
from typing import List, Dict


def _aggregate_stats(d, key, stats):

	if not isinstance(stats, np.ndarray):
		tmp_stats = np.array(stats)
	else:
		tmp_stats = stats

	_add_stats(d, key + " mean", tmp_stats.mean())
	_add_stats(d, key + " std", tmp_stats.std())
	_add_stats(d, key + " median", np.median(tmp_stats))


def _add_stats(d, k, v):
	if k not in d:
		d[k] = [v]
	else:
		d[k].append(v)


def collect_stats(*,
				  files,
				  headers=None,
				  **_):

	stats = {}
	columns = []

	for i, file in enumerate(files):

		p = Path(file)
		n = p.name

		if headers is None:
			columns.append(n)
		else:
			columns.append(headers[i])

		with open(p, "rb") as f:
			data = pickle.load(f)

		for k, v in data.items():
			if isinstance(v, (list, np.ndarray)):
				_aggregate_stats(stats, k, v)
			else:
				_add_stats(stats, k, v)

	return columns, stats


def print_stats(
		columns: List[str],
		stats: Dict, *,
		fmt="github",
		**_
	):

	headers = ["Stats"] + columns
	stats = [[k] + v for k, v in stats.items()]

	t = tabulate(stats, headers=headers, tablefmt=fmt)

	print(t)


def parse() -> Dict:
	parser = ArgumentParser()

	parser.add_argument("files", nargs="*", type=str)
	parser.add_argument("--headers", nargs="*", type=str, default=None)

	args = parser.parse_args()

	return vars(args)


def main():
	args = parse()
	c, s = collect_stats(**args)
	print_stats(c, s)


if __name__ == "__main__":
	main()
