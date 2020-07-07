import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default='/mnt/lustre/sjtu/home/zkz01/tools/CTC_pytorch/timit/data/train/wrd_text')
parser.add_argument("--outputlab", type=str, default='/mnt/lustre/sjtu/home/zkz01/tools/CTC_pytorch/timit/data/train/lab_char')
args = parser.parse_args()

wrd_file = open(args.text, 'r')

output_file = open(args.outputlab, 'w')

for line in wrd_file.readlines():
    line = line.rstrip("\n")
    index = line.split(" ")[0]
    trans = " ".join(line.split(" ")[1:])
    char_list = []
    for char in trans:
        if char == ' ':
            char_list.append("<eow>")
        else:
            char_list.append(char)
    trans_char = " ".join(char_list)
    output_file.write("{} {}\n".format(index, trans_char))

wrd_file.close()
output_file.close()