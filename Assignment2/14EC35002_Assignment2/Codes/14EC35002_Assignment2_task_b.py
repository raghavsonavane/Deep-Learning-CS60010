
# coding: utf-8

# In[2]:


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


# In[3]:


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
    


# # Input Data

# In[4]:


(train_images,train_labels)=dl.DataLoader().load_data('train')
(test_images,test_labels)=dl.DataLoader().load_data('test')
batch_size=32

ntrain=int(0.7*len(train_images))
train_iter = mx.io.NDArrayIter(train_images[:ntrain, :]/255, train_labels[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(train_images[ntrain:, :]/255, train_labels[ntrain:], batch_size)
test_iter = mx.io.NDArrayIter(test_images/255, test_labels, batch_size)

train_loader = DataIterLoader(train_iter)
val_loader = DataIterLoader(val_iter)
test_loader = DataIterLoader(test_iter)


# # Defining Neural Network

# In[5]:


net2 = nn.HybridSequential()

with net2.name_scope():
    net2.add(
        #nn.Dropout(.5)
        #nn.Flatten(),
        #nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(1024, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(512, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(256, activation='relu'),
        #nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net2.hybridize()


# In[6]:


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


# In[7]:


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


# ## Xavier Initialization

# In[8]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Xavier(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_xavier,val_loss_history_xavier,train_acc_history_xavier,val_acc_history_xavier = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_xavier.params'
net2.save_parameters(filename)
plt.plot(epoch_history, train_loss_history_xavier, epoch_history, val_loss_history_xavier )
plt.title('vanilla_net2_xavier_loss')
plt.savefig('vanilla_net2_xavier_loss.png')
plt.show()

plt.plot(epoch_history, train_acc_history_xavier, epoch_history, val_acc_history_xavier )
plt.title('vanilla_net2_xavier_accuarcy')
plt.savefig('vanilla_net2_xavier_accuarcy.png')
plt.show()


# ## Normal Initialization

# In[9]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_normal,val_loss_history_normal,train_acc_history_normal,val_acc_history_normal = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_normal.params'
net2.save_parameters(filename)
plt.plot(epoch_history, train_loss_history_normal, epoch_history, val_loss_history_normal)
plt.title('vanilla_net2_normal_loss')
plt.savefig('vanilla_net2_normal_loss.png')
plt.show()

plt.plot(epoch_history, train_acc_history_normal, epoch_history, val_acc_history_normal )
plt.title('vanilla_net2_normal_accuarcy')
plt.savefig('vanilla_net2_normal_accuarcy.png')
plt.show()


# ## Orthogonal Initialization

# In[10]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Orthogonal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_orthogonal,val_loss_history_orthogonal,train_acc_history_orthogonal,val_acc_history_orthogonal = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_orthogonal.params'
net2.save_parameters(filename)
plt.plot(epoch_history, train_loss_history_orthogonal, epoch_history, val_loss_history_orthogonal)
plt.title('vanilla_net2_orthogonal_loss')
plt.savefig('vanilla_net2_orthogonal_loss.png')
plt.show()

plt.plot(epoch_history, train_acc_history_orthogonal, epoch_history, val_acc_history_orthogonal )
plt.title('vanilla_net2_orthogonal_accuarcy')
plt.savefig('vanilla_net2_orthogonal_accuarcy.png')
plt.show()


# ## Experiment 1 Plots

# In[15]:


plt.plot(epoch_history, train_acc_history_xavier)
plt.plot(epoch_history, train_acc_history_normal)
plt.plot(epoch_history, train_acc_history_orthogonal)
plt.title('Training Accuracy with epochs')
plt.legend(('Xavier','Normal','Orthogonal'))
plt.savefig('exp1_training_acc.png')
plt.show()

plt.plot(epoch_history, train_loss_history_xavier)
plt.plot(epoch_history, train_loss_history_normal)
plt.plot(epoch_history, train_loss_history_orthogonal)
plt.title('Training Loss with epochs')
plt.legend(('Xavier','Normal','Orthogonal'))
plt.savefig('exp1_training_loss.png')
plt.show()

plt.plot(epoch_history, val_acc_history_xavier)
plt.plot(epoch_history, val_acc_history_normal)
plt.plot(epoch_history, val_acc_history_orthogonal)
plt.title('Validation Accuracy with epochs')
plt.legend(('Xavier','Normal','Orthogonal'))
plt.savefig('exp1_validation_acc.png')
plt.show()

plt.plot(epoch_history, val_loss_history_xavier)
plt.plot(epoch_history, train_loss_history_normal)
plt.plot(epoch_history, val_loss_history_orthogonal)
plt.title('Validation Loss with epochs')
plt.legend(('Xavier','Normal','Orthogonal'))
plt.savefig('exp1_validation_loss.png')
plt.show()


# ## Exp2 Batch Normalization 

# In[16]:


net2 = nn.HybridSequential()

with net2.name_scope():
    net2.add(
        #nn.Dropout(.5)
        #nn.Flatten(),
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(1024, activation='relu'),
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(512, activation='relu'),
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(256, activation='relu'),
        nn.BatchNorm(axis=1, center=True, scale=True),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net2.hybridize()


# In[17]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_batch_normal,val_loss_history_batch_normal,train_acc_history_batch_normal,val_acc_history_batch_normal = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_batch_normal.params'
net2.save_parameters(filename)

plt.plot(epoch_history, train_acc_history_normal)
plt.plot(epoch_history, train_acc_history_batch_normal)
plt.title('Training Accuracy with epochs')
plt.legend(('Without Batch Normalization','With Batch Normalization'))
plt.savefig('exp2_training_acc.png')
plt.show()

plt.plot(epoch_history, train_loss_history_normal)
plt.plot(epoch_history, train_loss_history_batch_normal)
plt.title('Training Loss with epochs')
plt.legend(('Without Batch Normalization','With Batch Normalization'))
plt.savefig('exp2_training_loss.png')
plt.show()


# ## Exp3 Dropout Comparison

# In[19]:


x = 0.6 #Dropout rate
net3 = nn.HybridSequential()

with net3.name_scope():
    net3.add(
        nn.Flatten(),
        nn.Dense(1024, activation='relu'),
        nn.Dropout(x),
        nn.Dense(512, activation='relu'),
        nn.Dropout(x),
        nn.Dense(256, activation='relu'),
        nn.Dropout(x),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net3.hybridize()

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net3.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer3 = gluon.Trainer(
    params=net3.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_dropout_06,val_loss_history_dropout_06,train_acc_history_dropout_06,val_acc_history_dropout_06 = neural_net_training(net3,trainer3,ctx)
neural_net_testing(net3,ctx)

filename='vanilla_net3_dropout_0.6_.params'
net3.save_parameters(filename)


# In[20]:


x = 0.4 #Dropout rate
net3 = nn.HybridSequential()

with net3.name_scope():
    net3.add(
        nn.Flatten(),
        nn.Dense(1024, activation='relu'),
        nn.Dropout(x),
        nn.Dense(512, activation='relu'),
        nn.Dropout(x),
        nn.Dense(256, activation='relu'),
        nn.Dropout(x),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net3.hybridize()

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net3.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer3 = gluon.Trainer(
    params=net3.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_dropout_04,val_loss_history_dropout_04,train_acc_history_dropout_04,val_acc_history_dropout_04 = neural_net_training(net3,trainer3,ctx)
neural_net_testing(net3,ctx)

filename='vanilla_net3_dropout_0.4_.params'
net3.save_parameters(filename)


# In[21]:


x = 0.1 #Dropout rate
net3 = nn.HybridSequential()

with net3.name_scope():
    net3.add(
        nn.Flatten(),
        nn.Dense(1024, activation='relu'),
        nn.Dropout(x),
        nn.Dense(512, activation='relu'),
        nn.Dropout(x),
        nn.Dense(256, activation='relu'),
        nn.Dropout(x),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net3.hybridize()

ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net3.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer3 = gluon.Trainer(
    params=net3.collect_params(),
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.04},
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_dropout_01,val_loss_history_dropout_01,train_acc_history_dropout_01,val_acc_history_dropout_01 = neural_net_training(net3,trainer3,ctx)
neural_net_testing(net3,ctx)

filename='vanilla_net3_dropout_0.1_.params'
net3.save_parameters(filename)


# In[22]:


plt.plot(epoch_history, train_acc_history_dropout_06)
plt.plot(epoch_history, train_acc_history_dropout_04)
plt.plot(epoch_history, train_acc_history_dropout_01)
plt.plot(epoch_history, train_acc_history_normal)
plt.title('Training Accuracy with epochs')
plt.legend(('Dropout 0.6','Dropout 0.4','Dropout 0.1','Vanilla Network'))
plt.savefig('exp3_training_acc.png')
plt.show()

plt.plot(epoch_history, train_loss_history_dropout_06)
plt.plot(epoch_history, train_loss_history_dropout_04)
plt.plot(epoch_history, train_loss_history_dropout_01)
plt.plot(epoch_history, train_loss_history_normal)
plt.title('Training Loss with epochs')
plt.legend(('Dropout 0.6','Dropout 0.4','Dropout 0.1','Vanilla Network'))
plt.savefig('exp3_training_loss.png')
plt.show()


# ## Exp 4: Optimizer comparison

# ## SGD Optimizer

# In[23]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='sgd',
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_sgd,val_loss_history_sgd,train_acc_history_sgd,val_acc_history_sgd = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_sgd.params'
net2.save_parameters(filename)


# ## Adam Optimizer 

# In[24]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='adam',
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_adam,val_loss_history_adam,train_acc_history_adam,val_acc_history_adam = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_adam.params'
net2.save_parameters(filename)


# ## AdaDelta

# In[27]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='AdaDelta',
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_AdaDelta,val_loss_history_AdaDelta,train_acc_history_AdaDelta,val_acc_history_AdaDelta = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_AdaDelta.params'
net2.save_parameters(filename)


# ## AdaGrad 

# In[26]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='AdaGrad',
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_AdaGrad,val_loss_history_AdaGrad,train_acc_history_AdaGrad,val_acc_history_AdaGrad = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_AdaGrad.params'
net2.save_parameters(filename)


# ## RMSProp 

# In[28]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='RMSProp',
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_RMSProp,val_loss_history_RMSProp,train_acc_history_RMSProp,val_acc_history_RMSProp = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_RMSProp.params'
net2.save_parameters(filename)


# ## Nesterov accelerated SGD

# In[29]:


ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)

net2.initialize(mx.init.Normal(),ctx=ctx,force_reinit=True)

trainer2 = gluon.Trainer(
    params=net2.collect_params(),
    optimizer='NAG',
)

# optimizer_paramsSoftmaxCrossEntropyLoss combines the softmax activation and the cross entropy loss 
# function in one layer, therefore the last layer in our network has no activation function.

metric = mx.metric.Accuracy()
loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
epoch_history,train_loss_history_NAG,val_loss_history_NAG,train_acc_history_NAG,val_acc_history_NAG = neural_net_training(net2,trainer2,ctx)
neural_net_testing(net2,ctx)

filename='vanilla_net2_NAG.params'
net2.save_parameters(filename)


# In[30]:


plt.plot(epoch_history, train_acc_history_sgd)
plt.plot(epoch_history, train_acc_history_adam)
plt.plot(epoch_history, train_acc_history_AdaDelta)
plt.plot(epoch_history, train_acc_history_AdaGrad)
plt.plot(epoch_history, train_acc_history_RMSProp)
plt.plot(epoch_history, train_acc_history_NAG)
plt.title('Training Accuracy with epochs')
plt.legend(('SGD','Adam','AdaDelta','AdaGrad','RMSProp','NAG'))
plt.savefig('exp4_training_acc.png')
plt.show()

plt.plot(epoch_history, train_loss_history_sgd)
plt.plot(epoch_history, train_loss_history_adam)
plt.plot(epoch_history, train_loss_history_AdaDelta)
plt.plot(epoch_history, train_loss_history_AdaGrad)
plt.plot(epoch_history, train_loss_history_RMSProp)
plt.plot(epoch_history, train_loss_history_NAG)
plt.title('Training Loss with epochs')
plt.legend(('SGD','Adam','AdaDelta','AdaGrad','RMSProp','NAG'))
plt.savefig('exp4_training_loss.png')
plt.show()

