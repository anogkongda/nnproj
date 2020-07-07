import argparse

parser =argparse.ArgumentParser()

parser.add_argument("--dir", type=str, default='data_multi/basque/test')
parser.add_argument("--num-kws",type=int, default=100)

args = parser.parse_args()

text = args.dir + "/lab.txt"
keywords = args.dir + "/keyword"
kw_lab = args.dir + "/kwlab.txt"

word_list = {}

# construct word list
with open(text, 'r') as f:
    for line in f.readlines():
        line = line.rstrip("\n")
        key = line.split(" ")[0]
        lab = "".join(line.split(" ")[1:]).split("<eow>")
        for word in lab:
            if word != '':
                if word in word_list.keys():
                    word_list[word] = word_list[word]+1
                else:
                    word_list[word] = 1

# get top num_kws keywords
keyword_list = []
for index,word in enumerate(sorted(word_list.items(),key=lambda item:item[1],reverse=True)):
    if index < args.num_kws:
        keyword_list.append(word[0])
    else:
        break

with open(keywords,"w") as f:
    for kw in keyword_list:
        f.write(kw+"\n")



with open(text, 'r') as f:
    kwlab = open(kw_lab,'w')
    for line in f.readlines():
        line = line.rstrip("\n")
        key = line.split(" ")[0]
        lab = "".join(line.split(" ")[1:]).split("<eow>")
        flags = []
        for kw in keyword_list:
            if kw in lab:
                flags.append(str(1))
            else:
                flags.append(str(0))
        kwlab.write("{} {}\n".format(key, " ".join(flags)))
    
    kwlab.close()
