import torch

import matplotlib.pyplot as plt

with open('datasets/train_glandular.csv','r') as f:
 lines=f.read()

lines=lines.split('\n')
lines = lines[1:-1]

with open('datasets/val_glandular.csv','r') as f:
 lines+=f.read().split('\n')[1:-1]

xys=[]
whs=[]
for l in lines:
 anns = l.split(',')[1]
 anns = anns.split(';')[:-1]
 for ann in anns:
  l, x1, y1, x2, y2 = ann.split(' ')
  whs.append([int(x2)-int(x1), int(y2)-int(y1)])
  xys.append([int(x2)+int(x1), int(y2)+int(y1)])

whs = torch.Tensor(whs)
xys = torch.Tensor(xys)/2

plt.scatter(whs[:,0],whs[:,1])
plt.xlabel('box width')
plt.ylabel('box height')
plt.show()