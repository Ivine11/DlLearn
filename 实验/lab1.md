lab2-1:

#### 全连接层：

输入一维度向量x， 维度m

输出一维向量y， 维度n

权重W二维矩阵m * n

偏置b n

$$y=W^T x + b​$$



##### 1.参数提取

需要两个参数 W， b，所以要参数初始话

##### 2.forward就是要对上面公式进行计算：

```python
        # TODO：全连接层的前向传播，计算输出结果
        self.output = np.dot(self.input, self.weight) + self.bias
```

##### 3.求梯度，也就是backward函数

完成一下几个公式：

![image-20220924134200680](/Users/pengruiying/Desktop/github/DlLearn/实验/assets/image-20220924134200680.png)

```python
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = top_diff
        bottom_diff = np.dot(top_diff, self.weight.T)
        return bottom_diff
```

##### 4.update_param(self, lr):  # 参数更新， 

就是使用已经求得的梯度，调整参数，lr可以看作调整幅度（因为要慢慢逼近，所以一次不能调大了

```python
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias
```



**实际：**

输入：batch_size * m

输出：batch_size * n

初始化：输入m, n

#### Relu

$$y(i) = max(0, x(i))$$

##### 1.上述公式对应的前向传播：

```python
        # TODO：ReLU层的前向传播，计算输出结果
        output = np.maximum(self.input,0)
```

##### 2.实验手册给出的求导公式：

![image-20220924134935171](/Users/pengruiying/Desktop/github/DlLearn/实验/assets/image-20220924134935171.png)

实现：

```python
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        b = self.input
        b[b>0] = 1
        b[b<0] = 0
        bottom_diff = np.multiply(b,top_diff) #np.dot是矩阵乘法，multiply是对应位置相乘
        return bottom_diff
```



#### Softmax:

$$y(i) = e^{x(i)}/ \sum_{j}e^{x(i)}$$

为了解决指数上溢出加入了max

##### 1.前向传播要实现的公式如下：

![image-20220924135457050](/Users/pengruiying/Desktop/github/DlLearn/实验/assets/image-20220924135457050.png)

```python
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
      #这里要全部按行，因为每行对应一组数据
        input_max = np.max(input, axis=1, keepdims=True) #按行求最大值
        input_exp = np.exp(input- input_max)#(100,10) #对每个x值分别求e^x
        # print(input_exp.shape)
        partsum = np.sum(input_exp, axis=1)#按行求最大值
        sum = np.tile(partsum,(input_exp.shape[1],1))#这里是因为按行求和之后，part_sum和input_exp的形状不一样，所以要对partsum进行复制
        self.prob = input_exp / sum.T
        return self.prob
```

##### 2.求导

![image-20220924140343349](/Users/pengruiying/Desktop/github/DlLearn/实验/assets/image-20220924140343349.png)

```python
        # TODO：softmax 损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff
```

##### 3.softmax是一个损失函数，所以这一层要求损失：

![image-20220924140503588](/Users/pengruiying/Desktop/github/DlLearn/实验/assets/image-20220924140503588.png)

```python
    def get_loss(self, label):   # 计算损失
    #self.prob是预测值，如果要预测3个类别，那么他的形状就是（3，）
    #如果真实类别是2，那么就要把label转成向量（0，1，0）也就是one_hot向量，得到self.label_onehot
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
```



