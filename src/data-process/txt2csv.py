
path = '../data/'

def txt_2_csv(f):
    fi = open(path + '{}.log.txt'.format(f), 'r')
    fo = open(path + '{}.csv'.format(f), 'w')

    for line in fi:
        line = line.replace(',', '')
        line = line.replace('\t', ',')
        fo.write(line)
    fi.close()
    fo.close()

if __name__=='__main__':
    fs = ['train', 'test']
    for f in fs:
        txt_2_csv(f)
        print(f + ' completed!')

