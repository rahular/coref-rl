import sys
import json

def create_json(csv_path):
    infile = open(csv_path, 'r', encoding='utf-8')
    outfile = open(csv_path.split('.')[0] + '.json', 'w', encoding='utf-8')
    d = {'muc':[], 'b_cubed':[], 'ceafe':[]}

    lines = infile.readlines()
    for idx, line in enumerate(lines):
        nums = [float(num) for num in line.strip().split(',')]
        if idx % 3 == 0:
            d['muc'].append(nums)
        elif idx % 3 == 1:
            d['b_cubed'].append(nums)
        elif idx % 3 == 2:
            d['ceafe'].append(nums)

    assert len(d['muc']) == len(d['b_cubed']) == len(d['ceafe']) == 348
    json.dump(d, outfile)

    outfile.close()
    infile.close()
        
if __name__ == '__main__':
    create_json(sys.argv[1])