import os

def write_name(np, tx):
    # npz文件路径
    files = os.listdir(np)
    # txt文件路径
    f = open(tx, 'w')
    for i in files:
        name = i[:-4] + '\n'
        f.write(name)

write_name('./data/swinyseg_new/test_vol_h5/', './lists/lists_swinyseg_new/test_vol.txt')