import os

data = {}

with open('labels_3.csv', 'r') as f:
    for line in f:
        spl = line.split(',')
        if len(spl) > 1:
            # bird,129,186,164,103,pos/image_2020-03-26_12.39.16_045.jpg,640,480
            # label Xl Yt Xw Yh path Xim Yim
            # We want:
            # path count Xm Ym Xw Yh    Xm2 Ym2 Xw2 Yh2
            path = spl[5]
            if not path.startswith('pos'): path = 'pos/'+path
            x = max(int(spl[1]), 0)
            y = spl[2]
            w = spl[3]
            h = spl[4]
            if path not in data:
                data[path] = [path, 1, x, y, w, h]
            else:
                data[path][1] += 1
                data[path].extend([x, y, w, h])

with open('transformed_labels_3.txt', 'w') as f:
    for d in data:
        f.write(' '.join(str(x) for x in data[d]) + '\n')