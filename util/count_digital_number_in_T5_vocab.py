

from transformers import T5Tokenizer
model_version = 't5-11b'
tokenizer = T5Tokenizer.from_pretrained(model_version)
vocab_list = tokenizer.get_vocab()

count = 0
for i in list(vocab_list.keys()):
    if i.isdigit():
        count += 1
        # print(i)

print(count, len(list(vocab_list.keys())))
# print(list(vocab_list.keys())[:100])