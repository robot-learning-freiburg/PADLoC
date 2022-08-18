import pickle
import os

def out2pickle(filename):
    old_filename = None
    ap_fp = None
    ap_fn = None

    with open(filename, "r") as f:
        for l in f:
            if old_filename and ap_fp and ap_fn:
                break

            if "Saving pairwise distances to" in l:
                old_filename = l.split(" ")[-1][:-2].split("/")[-1]
                continue

            if "AP FP:" in l:
                ap_fp = float(l.split(" ")[-1])
                continue

            if "AP FN:" in l:
                ap_fn = float(l.split(" ")[-1])
                continue

    if not(old_filename and ap_fp and ap_fn):
        return

    new_filename = old_filename.split(".")[0] + ".pickle"
    res = {
        "AP_FP": ap_fp,
        "AP_FN": ap_fn
    }

    print(f"Saving stats to {new_filename}.")
    with open(new_filename, "wb") as f:
        pickle.dump(res, f)


def main(path="./"):
    files = sorted(os.listdir(path))

    for file in files:
        if ".out" in file:
            out2pickle(file)


if __name__ == "__main__":
    main()
