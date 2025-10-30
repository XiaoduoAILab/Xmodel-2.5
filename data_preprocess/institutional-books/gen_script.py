with open('file_list.txt', 'r') as fp:
    lines = fp.readlines()

template = 'nohup python data_preprocess/institutional-books/to_json.py --data_path /data12/datasets/institutional-books-1.0/data/{name} &'

with open('to_json.sh', 'w') as fp:
    for i in range(len(lines)):
        line = lines[i]
        name = line.replace('\n', '')
        cmd = template.format(name=name)

        fp.write(cmd + '\n')

        if i > 0 and i % 100 == 0:
            fp.write('\nsleep 1m\n\n')




