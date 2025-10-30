with open('file_list.txt', 'r') as fp:
    lines = fp.readlines()

template = 'nohup python tools/preprocess_data.py --input /data6/datasets/dolma/target/{name}.json --output-prefix /data9/datasets/pretrain/dolma_wo_cc/{name} --tokenizer-model tokenizers/deepseekv3 --tokenizer-type HuggingFaceTokenizer --append-eod --json-keys content --workers 2 &'

with open('prepare.sh', 'w') as fp:

    for i in range(len(lines)):
        line = lines[i]
        name = line.replace('.json\n', '')
        cmd = template.format(name=name)

        fp.write(cmd + '\n')

        if i > 0 and i % 100 == 0:
            fp.write('\nsleep 15m\n\n')

        


    