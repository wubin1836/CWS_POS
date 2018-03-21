word = {}
max_target = 0
target_len = {}

with open("train/source.txt") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split()
        if max_target < len(tmp):
            max_target = len(tmp)

        if target_len.has_key(len(tmp)):
            target_len[len(tmp)] += 1
        else:
            target_len[len(tmp)] = 1
        for item in tmp:
            if not word.has_key(item):
                word[item] = len(word)

# with open("dev/source.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         tmp = line.split()
#         if max_target < len(tmp):
#             max_target = len(tmp)
#         for item in tmp:
#             if not word.has_key(item):
#                 word[item] = len(word)
#
# with open("test/source.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         tmp = line.split()
#         if max_target < len(tmp):
#             max_target = len(tmp)
#         for item in tmp:
#             if not word.has_key(item):
#                 word[item] = len(word)

print target_len

print max_target
# print len(word)
# import json
# with open('label2idx.json', 'w') as fw:
#     json.dump(word, fw)