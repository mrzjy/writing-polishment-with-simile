import argparse


def diversity(file):
    def compute(alist):
        return len(set(alist)) / len(alist)

    with open(file, "r") as f:
        lines = [l.strip() for l in f.readlines()]

        ngrams1, ngrams2 = [], []
        for l in lines:
            tokens = l.split(" ")
            ngrams1.extend(tokens)
            ngrams2.extend(["".join(tokens[i:i + 2]) for i in range(0, len(tokens) - 1)])

    return compute(ngrams1), compute(ngrams2), compute(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', help="predicted text file, one example per line")
    args = parser.parse_args()

    dist1, dist2, distS = diversity(args.f)
    print("{}: Diversity\t{:.3f}\t{:.3f}\t{:.3f}".format(args.f, dist1, dist2, distS))
