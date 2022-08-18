#!/usr/bin/env python

from argparse import ArgumentParser
import pickle


def main(filename):
    with open(filename, "rb") as f:
        dat = pickle.load(f)

    print("{")
    for k, v in dat.items():
        print(f"\t{k}:\t{v}")
    print("}")


def cli_args():
    parser = ArgumentParser()

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    main(**cli_args())
