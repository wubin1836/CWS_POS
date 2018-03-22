word = {}
max_target = 0
target_len = {}

fw = open("test/target_s.txt", "w")
with open("test/target.txt") as f:
    lines = f.readlines()
    for line in lines:
        tmp = line.split()
        ll = ""
        for item in tmp:
            ll = ll + item.split("_")[0] + " "
        fw.write(ll.strip() + "\n")
