

from transformers import T5Tokenizer
model_version = 't5-11b'
tokenizer = T5Tokenizer.from_pretrained(model_version)
vocab_list = tokenizer.get_vocab()

count = 0
s = set()
count_ = 0
s_ = set()
for i in list(vocab_list.keys()):

    if i.isdigit():
        count += 1
        s.add(i)
    else:
        i_s = i.split('▁')
        if len(i_s) > 1 and i_s[1].isdigit():
            count_ += 1
            s_.add(i_s[1])

print(count, count_, len(list(vocab_list.keys())))
s_2 = s.union(s_)
print(len(s_2), s_2)
# print(list(vocab_list.keys())[:100])