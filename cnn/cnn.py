import torch.nn as nn


class CNN(nn.Module):
    def __init__(self,conv1_out=2,conv2_out=8):
        super(CNN, self).__init__()

        self.conv1_out = conv1_out
        self.conv2_out = conv2_out

        self.conv1 = nn.Sequential(         # 输入大小 (1, 8, 12)
            nn.Conv2d(
                in_channels=1,              # 灰度图
                out_channels=self.conv1_out,            # 要得到几多少个特征图
                kernel_size=3,              # 卷积核大小
                stride=1,                   # 步长
                padding=1,                  # 如果希望卷积后大小跟原来一样，需要设置padding=(kernel_size-1)/2 if stride=1
            ),                              # 输出的特征图为 (16, 28, 28)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(kernel_size=2),    # 进行池化操作（2x2 区域）, 输出结果为： (16, 14, 14)
        )
        #self.dropout = nn.Dropout(args.dropout)
        self.conv2 = nn.Sequential(         # 下一个套餐的输入 (16, 14, 14)
            nn.Conv2d(self.conv1_out,self.conv2_out, 3, 1, 1),     # 输出 (32, 14, 14)
            nn.ReLU(),                      # relu层
            nn.MaxPool2d(2),                # 输出 (32, 7, 7)
        )
        self.out = nn.Linear(self.conv2_out * 46 * 3, 184*3)   # 全连接层得到的结果

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten操作，结果为：(batch_size, 32 * 7 * 7)
        #print(x.shape)
        output = self.out(x)
        return output