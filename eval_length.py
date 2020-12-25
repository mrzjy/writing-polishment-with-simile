import argparse


def length(file):
    with open(file, "r") as f:
        lines = [len(l.strip().split()) for l in f.readlines()]

    return sum(lines) / len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', help="predicted text file, one example per line")
    args = parser.parse_args()

    l = length(args.f)
    print("{}: avg_Length\t{:.3f}".format(args.f, l))
