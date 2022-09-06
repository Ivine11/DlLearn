import numpy as np


class FullyConnectedLayer(object):
    num_input = None
    num_output = None
    weight = None
    bias = None
    input = None
    output = None

    # 全连接层初始化
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output

    # 参数初始化
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])

    # 前向传播计算
    def forward(self, input):
        self.input = input
        # 点乘
        self.output = np.dot(self.input, self.weight) + self.bias
        return self.output

    def backward(self, top_diff):
        # TODO:全连接层参数利用梯度进行更新
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = top_diff
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff

    def update_param(self, lr):
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def save_param(self):
        return self.weight, self.bias


class ReLULayer(object):
    def forward(self, input):
        self.input = input
        output = np.maximum(self.input, 0)
        return output

    def backward(self, top_diff):
        b = self.input
        b[b > 0] = 1
        b[b < 0] = 0
        bottom_diff = np.multiply(top_diff, b)
        return bottom_diff


class SoftmaxLossLayer(object):
    y = None
    one_hot = None
    batch_size = None

    def forward(self, x):
        # 计算每行最大值 一行是一个因变量
        row_max = np.max(x)
        # 每行元素减去最大值否则会溢出
        x = x - row_max
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1)  # 求每一列的和
        # 按轴复制， np(a, (2, 1))第一个y轴复制倍数， 第二个x轴复制倍数
        x_sum = np.tile(x_sum, (x_exp.shape[1], 1))
        # print(x_sum)
        self.y = x_exp / x_sum.T
        # print("x_exp:",x_exp,"\nx_sum:", x_sum, "\ny:", self.y )
        return self.y

    def get_loss(self, label):
        self.batch_size = self.y.shape[0]
        self.one_hot = np.zeros_like(self.y)
        self.one_hot[np.arange(self.batch_size), label]= 1
        soft_loss = (-np.log(self.y) * self.one_hot).sum()
        return soft_loss
        # return 0

    def backward(self):
        p = self.batch_size
        print("样本数量：", p)
        bottom_diff = (self.y - self.one_hot) / p
        return bottom_diff


class MNIST_MLP(object):
    def __init__(self, batch_size=100, input_size=784, hidden1=32, hidden2=64, out_classes=10, lr=0.01, max_epoch=2, print_iter=100):
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.out_classes = out_classes
        self.lr = lr
        self.max_epoch = max_epoch
        self.print_iter = print_iter

    def build_model(self):
        # TODO:建立三层神经网络
        self.fc1 = FullyConnectedLayer(self.input_size, self.hidden1)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(self.hidden1, self.hidden2)
        self.relu2 = ReLULayer()
        self.fc3 = FullyConnectedLayer(self.hidden2, self.out_classes)
        self.softmax = SoftmaxLossLayer()
        self.update_layer_list = [self.fc1, self.fc2, self.fc3]

    def init_model(self):
        for layer in self.update_layer_list:
            layer.init_param()

    def forward(self, input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        h2 = self.relu2.forward(h2)
        h3 = self.fc3.forward(h2)
        self.prob = self.softmax.forward(h3)
        return self.prob

    def backward(self):
        dloss = self.softmax.backward()
        dh3 = self.fc3.backward(dloss)
        dh2 = self.relu2.backward(dh3)
        dh2 = self.fc2.backward(dh2)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer_list:
            layer.update_param(lr)

    def save_model(self, param_dir):
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        params['w3'], params['b3'] = self.fc3.save_param()
        np.save(param_dir, params)

    def load_model(self, param_dir):
        params = np.load(param_dir).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'])



    # def train(self):
    #     max_batch = self.train_data.shape[0] / self.batch_size
    #     for idx_epoch in range(self.max_epoch):
    #         mlp.shuffle_data()
    #         for idx_batch in range(max_batch):
    #














if __name__ == '__main__':
    # FullLay = FullyConnectedLayer(4, 2)
    # FullLay.init_param()
    # print("FullLay.weight.shape", FullLay.weight.shape)
    # print(FullLay.weight)
    # print(" FullLay.bias.shape:", FullLay.bias.shape)
    # print(FullLay.bias)
    '''
    FullLay.weight.shape (4, 2)
        [[-0.01269196  0.002817  ]
         [ 0.0001468  -0.00751825]
         [ 0.00977201  0.00285932]
         [ 0.00811444  0.02674567]]
     FullLay.bias.shape: (1, 2)
        [[0. 0.]]
    '''
    x = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    soft = SoftmaxLossLayer()
    soft.forward(x)
    print("y:", soft.y)
    loss = soft.get_loss([1, 2])
    print("one-hot_label:", soft.one_hot)
    print("loss:", loss)
    '''
    y: [[0.00862189 0.02343672 0.0637076  0.17317522]
     [0.02343672 0.0637076  0.17317522 0.47073904]]
    one-hot_label: [[0. 1. 0. 0.]
     [0. 0. 1. 0.]]
    loss: 5.506902772158837
    '''
    a = soft.backward()
    print(a)