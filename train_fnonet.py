import json
import pickle
from timeit import default_timer

import numpy as np
import torch.optim.lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.model.FNONet import FNONet
from src.utils.Adam import Adam
from src.utils.utils import *

train_dataset=pickle.load(open("./data/train_dataset.pkl","rb"))
test_dataset=pickle.load(open("./data/test_dataset.pkl","rb"))
y=pickle.load(open('data/dataY.pkl','rb'))
y = torch.from_numpy(y).float()
batch = y.shape[0]
nx = y.shape[2]  # 172
ny = y.shape[3]  # 79
channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1)
        .reshape((batch*nx*ny,3)) ** 2, dim=0)).view(1, -1, 1, 1).cuda()
batch_size=20
epochs=1000
modes1=20
modes2=20
width=32
learning_rate=0.001
step_size=50
gamma=0.9
ntrain=len(train_dataset)
ntest=len(test_dataset)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model=FNONet(modes1,modes2,width).cuda()

myloss=LpLoss()

myloss_rel=LpLoss_rel(size_average=False)
optim = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

scheduler=torch.optim.lr_scheduler.StepLR(optim,step_size,gamma)

train_mse_list = []
train_l2_rel_list = []
test_l2_rel_list = []
test_mse_list = []

train_mse_total_list = []
train_mse_ux_list = []
train_mse_uy_list = []
train_mse_p_list = []

test_mse_total_list = []
test_mse_ux_list = []
test_mse_uy_list = []
test_mse_p_list = []



best_test = float('inf')
best_epoch=0
best_model_path = 'results/best_model.pth'
best_data={}
for epoch in range(1,epochs+1):
    model.train()
    t_start=default_timer()
    train_mse = 0
    train_l2_rel = 0

    train_mse_total=0
    train_mse_ux=0
    train_mse_uy=0
    train_mse_p=0

    test_mse_total=0
    test_mse_ux=0
    test_mse_uy=0
    test_mse_p=0

    for train_x,train_y in train_loader:
        train_x,train_y=train_x.cuda(),train_y.cuda()
        optim.zero_grad()
        out=model(train_x)

        loss = myloss(out,train_y,channels_weights)
        mse = F.mse_loss(out, train_y, reduction='mean')

        l2_rel=myloss_rel(out,train_y)

        loss.backward()
        optim.step()

        train_mse += mse.item()
        train_l2_rel += l2_rel.item()

        mse_total = float(torch.sum((out - train_y) ** 2))
        mse_ux = float(torch.sum((out[:, 0, :, :] - train_y[:, 0, :, :]) ** 2))
        mse_uy = float(torch.sum((out[:, 1, :, :] - train_y[:, 1, :, :]) ** 2))
        mse_p = float(torch.sum((out[:, 2, :, :] - train_y[:, 2, :, :]) ** 2))
        train_mse_total+=mse_total
        train_mse_ux+=mse_ux
        train_mse_uy+=mse_uy
        train_mse_p+=mse_p
    train_mse/=len(train_loader)
    train_l2_rel /= ntrain

    train_mse_list.append(train_mse)
    train_l2_rel_list.append(train_l2_rel)

    train_mse_total/=ntrain
    train_mse_ux/=ntrain
    train_mse_uy/=ntrain
    train_mse_p/=ntrain
    train_mse_total_list.append(train_mse_total)
    train_mse_ux_list.append(train_mse_ux)
    train_mse_uy_list.append(train_mse_uy)
    train_mse_p_list.append(train_mse_p)

    scheduler.step()
    model.eval()
    test_l2_rel=0
    test_mse=0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.cuda(), test_y.cuda()

            out = model(test_x)
            mse=F.mse_loss(out,test_y, reduction='mean')
            mse_total = float(torch.sum((out - test_y) ** 2))
            mse_ux = float(torch.sum((out[:, 0, :, :] - test_y[:, 0, :, :]) ** 2))
            mse_uy = float(torch.sum((out[:, 1, :, :] - test_y[:, 1, :, :]) ** 2))
            mse_p = float(torch.sum((out[:, 2, :, :] - test_y[:, 2, :, :]) ** 2))

            test_l2=myloss_rel(out,test_y)
            test_mse+=mse.item()
            test_l2_rel+=test_l2.item()
            test_mse_total+=mse_total
            test_mse_ux+=mse_ux
            test_mse_uy+=mse_uy
            test_mse_p+=mse_p
    test_l2_rel/=ntest
    test_mse/=len(test_loader)
    test_mse_list.append(test_mse)
    test_l2_rel_list.append(test_l2_rel)

    test_mse_total/=ntest
    test_mse_ux /= ntest
    test_mse_uy /= ntest
    test_mse_p /= ntest

    test_mse_total_list.append(test_mse_total)
    test_mse_ux_list.append(test_mse_ux)
    test_mse_uy_list.append(test_mse_uy)
    test_mse_p_list.append(test_mse_p)

    if test_mse_total < best_test:
        best_test = test_mse_total
        best_epoch=epoch
        torch.save(model.state_dict(), best_model_path)

    t_end = default_timer()
    print(epoch)
    print(t_end - t_start)
    print(train_mse, test_mse)
    print(train_l2_rel, test_l2_rel)
    print(train_mse_total,train_mse_ux,train_mse_uy,train_mse_p)
    print(test_mse_total,test_mse_ux,test_mse_uy,test_mse_p)

    if epoch == best_epoch:

        best_data['best_epoch']=best_epoch
        best_data['train_mse']=train_mse
        best_data['test_mse']=test_mse
        best_data['train_l2_rel']=train_l2_rel
        best_data['test_l2_rel']=test_l2_rel
        best_data['train_mse_total']=train_mse_total
        best_data['test_mse_total']=test_mse_total
        best_data['train_mse_ux']=train_mse_ux
        best_data['test_mse_ux']=test_mse_ux
        best_data['train_mse_uy']=train_mse_uy
        best_data['test_mse_uy']=test_mse_uy
        best_data['train_mse_p']=train_mse_p
        best_data['test_mse_p']=test_mse_p

print("Best_metrics")
print(best_data)
with open('results/best_metrics_7.json', 'w') as f:
    json.dump(best_data, f)
loss_data = {
    'train_mse_list': train_mse_list,
    'train_l2_rel_list': train_l2_rel_list,
    'test_l2_rel_list': test_l2_rel_list,
    'test_mse_list': test_mse_list,
    'train_mse_total_list': train_mse_total_list,
    'train_mse_ux_list': train_mse_ux_list,
    'train_mse_uy_list': train_mse_uy_list,
    'train_mse_p_list': train_mse_p_list,
    'test_mse_total_list': test_mse_total_list,
    'test_mse_ux_list': test_mse_ux_list,
    'test_mse_uy_list': test_mse_uy_list,
    'test_mse_p_list': test_mse_p_list
}

with open('results/loss_data_7.json', 'w') as f:
    json.dump(loss_data, f)

def plot_loss(train_loss, test_loss, title):
    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

plot_loss(train_mse_list, test_mse_list, 'Train and Test MSE Loss')
plot_loss(train_l2_rel_list, test_l2_rel_list, 'Train and Test L2 Rel Loss')
plot_loss(train_mse_total_list, test_mse_total_list, 'Train and Test Total MSE Loss')
plot_loss(train_mse_ux_list, test_mse_ux_list, 'Train and Test UX MSE Loss')
plot_loss(train_mse_uy_list, test_mse_uy_list, 'Train and Test UY MSE Loss')
plot_loss(train_mse_p_list, test_mse_p_list, 'Train and Test P MSE Loss')


