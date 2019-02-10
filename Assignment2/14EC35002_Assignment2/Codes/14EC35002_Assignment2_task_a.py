
# coding: utf-8

# In[35]:


import argparse
import numpy as np
import mxnet as mx
import data_loader as dl
import module
import os
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from multiprocessing import cpu_count
mx.random.seed(42)
import matplotlib.pyplot as plt

#os.listdir()


# In[22]:


def transform(data, label):
    data = data.astype('float32')/255
    return data, label

def shuffle_dataset(X,Y):
    
    '''
        Write code to shuffle the dataset here. 
        
        Args: 
            X: Input feature ndarray
            Y: Input values ndarray
            
        Return:
            X and Y shuffled in place
    
    '''
    r = np.arange(len(X))
    np.random.shuffle(r)
    X = X[r]
    Y = Y[r]
    return (X,Y)
    pass

class DataIterLoader():
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        assert len(batch.data) == len(batch.label) == 1
        data = batch.data[0]
        label = batch.label[0]
        return data, label
    


# In[23]:


(train_images,train_labels)=dl.DataLoader().load_data('train')
(test_images,test_labels)=dl.DataLoader().load_data('test')
batch_size=32

ntrain=int(0.7*len(train_images))
#Mapping 0-255 intensity scale to 0-1 (without batch normalization)
train_iter = mx.io.NDArrayIter(train_images[:ntrain, :]/255, train_labels[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(train_images[ntrain:, :]/255, train_labels[ntrain:], batch_size)
test_iter = mx.io.NDArrayIter(test_images/255, test_labels, batch_size)

train_loader = DataIterLoader(train_iter)
val_loader = DataIterLoader(val_iter)
test_loader = DataIterLoader(test_iter)


# In[24]:


# data_loader=mx.gluon.data.dataset.ArrayDataset(X_train,Y_train)
# train_loader=mx.gluon.data.DataLoader(train_loader, batch_size=5, num_workers=cpu_count())

net1 = nn.HybridSequential()

with net1.name_scope():
    net1.add(
        #nn.BatchNorm(axis=1, center=True, scale=True)
        nn.Dense(512, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True)
        nn.Dense(128, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True)
        nn.Dense(64, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True)
        nn.Dense(32, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True)
        nn.Dense(16, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True)
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net1.hybridize()


# In[25]:


net2 = nn.HybridSequential()

with net2.name_scope():
    net2.add(
        nn.Dense(1024, activation='relu'),
        nn.Dense(512, activation='relu'),
        nn.Dense(256, activation='relu'),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net2.hybridize()


# In[28]:


def neural_net_training(net,trainer,ctx):
    epochs = 10
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    epoch_history = []
    for epoch in range(epochs):
        # training loop (with autograd and trainer steps, etc.)
        cumulative_train_loss = mx.nd.zeros(1, ctx=ctx)
        training_samples = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = loss_function(output, label)
            loss.backward()
            metric.update(label, output)
            trainer.step(data.shape[0])
            cumulative_train_loss += loss.sum()
            training_samples += data.shape[0]


        train_loss = cumulative_train_loss.asscalar()/training_samples
        train_name, train_acc = metric.get()
        #print('After epoch {}: {} = {}'.format(epoch + 1, name, acc))
        metric.reset()

        # validation loop
        cumulative_val_loss = mx.nd.zeros(1, ctx)
        val_samples = 0
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
            label = label.as_in_context(ctx)
            output = net(data)
            loss = loss_function(output, label)
            cumulative_val_loss += loss.sum()
            val_samples += data.shape[0]
            metric.update(label, output)
        val_loss = cumulative_val_loss.asscalar()/val_samples
        val_name, val_acc = metric.get()
        metric.reset()

        print("Epoch {}, training loss: {:.2f}, validation loss: {:.2f}".format(epoch, train_loss, val_loss))
        print("training accuracy: {}, validation accuracy: {}".format(train_acc,val_acc))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        epoch_history.append(epoch+1)
#     plt.plot(epoch_history, train_loss_history, epoch_history, val_loss_history )
#     plt.title('Loss')
#     plt.show()

#     plt.plot(epoch_history, train_acc_history, epoch_history, val_acc_history )
#     plt.title('Accuracy')
#     plt.show()
    return epoch_history,train_loss_history,val_loss_history,train_acc_history,val_acc_history

    pass


# In[29]:


def neural_net_testing(net,ctx):
    cumulative_test_loss = mx.nd.zeros(1, ctx)
    test_samples = 0
    for batch_idx, (data, label) in enumerate(test_loader):
            data = data.as_in_context(ctx).reshape((-1, 784)) # 28*28=784
            label = label.as_in_context(ctx)
            output = net(data)
            loss = loss_function(output, label)
            cumulative_test_loss += loss.sum()
            test_samples += data.shape[0]
            metric.update(label, output)
    test_loss = cumulative_test_loss.asscalar()/test_samples
    test_name, test_acc = metric.get()
    print("Final testing loss: {:.2f}".format(test_loss))
    print("testing accuracy: {}".format(test_acc))
    pass


# In[30]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net1.initialize(mx.init.Uniform(),ctx=ctx,force_reinit=True)
trainer1 = gluon.Trainer(
    params=net1.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history,val_loss_history,train_acc_history,val_acc_history = neural_net_training(net1,trainer1,ctx)
neural_net_testing(net1,ctx)


# In[31]:


filename='vanilla_net1.params'
net1.save_parameters(filename)
plt.plot(epoch_history, train_loss_history, epoch_history, val_loss_history )
plt.title('vanilla_net1_loss')
plt.savefig('vanilla_net1_loss.png')
plt.show()

plt.plot(epoch_history, train_acc_history, epoch_history, val_acc_history )
plt.title('vanilla_net1_accuarcy')
plt.savefig('vanilla_net1_accuarcy.png')
plt.show()


# In[32]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Uniform(),ctx=ctx,force_reinit=True)
trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history,val_loss_history,train_acc_history,val_acc_history = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)


# In[37]:


filename='vanilla_net2'
net2.export(filename)
plt.plot(epoch_history, train_loss_history, epoch_history, val_loss_history )
plt.title('vanilla_net2 Loss')
plt.savefig('vanilla_net2_loss.png')
plt.show()

plt.plot(epoch_history, train_acc_history, epoch_history, val_acc_history )
plt.title('vanilla_net2 Accuracy')
plt.savefig('vanilla_net2_accuarcy.png')
plt.show()

