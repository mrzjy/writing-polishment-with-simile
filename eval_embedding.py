import argparse
import pickle
import os

# import gensim

from eval_utils import average, greedy_match, extrema_score, extract_pretrained_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', help="ground truth text file, one example per line")
    parser.add_argument('--out', help="predicted text file, one example per line")
    parser.add_argument('--embeddings', default="embeddings.pkl", help="embeddings bin file")
    parser.add_argument('--gensim', action="store_true", required=False, help='from gensim or not')
    parser.add_argument('--vocab_file', default="data/vocab_seg_as_jieba.txt", help="embeddings bin file")
    args = parser.parse_args()

    if not args.gensim:
        load_fn = lambda file: pickle.load(open(file, "rb"))
    else:
        pass
        # load_fn = lambda file: gensim.models.Word2Vec.load(file)
    if os.path.exists(args.embeddings):
        print("loading embeddings file...")
        w2v = load_fn(args.embeddings)
    else:
        print("extract embeddings")
        extract_pretrained_embedding("Tencent_AILab_ChineseEmbedding.txt", args.vocab_file,
                                     embedding_dim=200, output_file=args.embeddings)
        w2v = load_fn(args.embeddings)

    r = average(args.ref, args.out, w2v)
    em = r[0]
    print("Embedding Average Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    r = greedy_match(args.ref, args.out, w2v)
    gm = r[0]
    print("Greedy Matching Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    r = extrema_score(args.ref, args.out, w2v)
    ve = r[0]
    print("Extrema Score: %f +/- %f ( %f )" % (r[0], r[1], r[2]))

    print("{:.3f}\t{:.3f}\t{:.3f}".format(em, gm, ve))
