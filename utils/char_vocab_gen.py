import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default='/mnt/lustre/sjtu/home/zkz01/wakeup/data/feats/voxforge/Catalan/text')
parser.add_argument("--vocab", type=str, default='/mnt/lustre/sjtu/home/zkz01/wakeup/data/feats/voxforge/Catalan/vocab')

args = parser.parse_args()

text_file = open(args.text, 'r')
vocab_file = open(args.vocab, 'w')
vocab = set()

# input the text file and generate the vocab file.

for line in text_file.readlines():
    trans = " ".join(line.split(" ")[1:]).rstrip("\n")
    for grapheme in trans:
        if not grapheme in vocab:
            vocab.add(grapheme)

vocab.remove(" ")
vocab.add("<eow>")

for index,grapheme in enumerate(vocab):
    vocab_file.write(grapheme + " " + "\n")

text_file.close()
vocab_file.close()
