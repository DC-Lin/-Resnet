import torch
from torch import nn,optim
import visdom
from torch.utils.data import DataLoader
import torchvision
from dataload import NumbersDataset
from torch.nn import functional as F
batchsize=32
lr=1e-3
epochs=10
device=torch.device('cpu')
torch.manual_seed(12345)
train_pokemon=NumbersDataset('pokeman',224,mode='train')
val_pokemon=NumbersDataset('pokeman',224,mode='val')
test_pokemon=NumbersDataset('pokeman',224,mode='test')
train_loader=DataLoader(train_pokemon,batch_size=batchsize,shuffle=True,num_workers=4)
val_loader=DataLoader(val_pokemon,batch_size=batchsize,num_workers=4)
test_loader=DataLoader(test_pokemon,batch_size=batchsize,num_workers=4)
viz=visdom.Visdom()
class ResNet18(nn.Module):
    def __init__(self,num_class):
        super(ResNet18,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(64),
        )
        self.block1=ResNet(64,128,stride=2)
        self.block2=ResNet(128,256,stride=2)
        self.block3=ResNet(256,512,stride=2)
        self.block4=ResNet(512,1024,stride=2)
        self.outlayer = nn.Linear(1024 * 7* 7, num_class)

    def forward(self, x):
        '''

        :param x:
        :return:
        '''
        x=F.relu(self.conv1(x))
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        # print(x.shape)
        x=x.view(x.size(0),-1)
        x=self.outlayer(x)
        return x

class ResNet(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(ResNet,self).__init__()
        self.conv1=nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(ch_out)
        self.conv2=nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(ch_out)
        #,ch_in and ch_out maybe different
        self.extra=nn.Sequential()
        if ch_out != ch_in:
            self.extra=nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        '''
        :param x:[b,ch,h,w]
        :return ch_out:
        '''
        x_out=F.relu(self.bn1(self.conv1(x)))
        x_out=self.bn2(self.conv2(x_out))
        #resnet short cut
        x_out=self.extra(x)+x_out
        x_out=F.relu(x_out)
        return x_out
def evalute(model,loader):
    correct=0
    total=len(loader.dataset)
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.no_grad():
            logits=model(x)
            pred=logits.argmax(dim=1)
        correct+=torch.eq(pred,y).sum().float().item()
    return correct/total
def main():
    model=ResNet18(5).to(device)
    optimizer=optim.Adam(model.parameters(),lr=lr)
    criteon=nn.CrossEntropyLoss()
    best_acc,best_epoch=0,0
    viz.line([0],[0],win='loss',opts=dict(title='loss'))
    viz.line([0],[0],win='val_acc',opts=dict(title='val_acc'))
    global_step=0
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_loader):
            #x:[b,3,224,224],y:[b]
            x,y=x.to(device),y.to(device)
            logits=model(x)
            loss=criteon(logits,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step+=1
        if epochs%2==0:
            val_acc=evalute(model,val_loader)
            if val_acc>best_acc:
                best_epoch=epoch
                best_acc=val_acc
                torch.save(model.state_dict(),'best.mdl')
                viz.line([val_acc], [global_step], win='val_acc', update='append')
    print('best acc',best_acc,'best epoch',best_epoch)
    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')
    test_acc=evalute(model,test_loader)
    print('test_acc',test_acc)

if __name__ == '__main__':
    main()