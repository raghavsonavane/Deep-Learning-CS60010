
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


# In[4]:


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
    


# In[6]:


(train_images,train_labels)=dl.DataLoader().load_data('train')
(test_images,test_labels)=dl.DataLoader().load_data('test')
batch_size=32

ntrain=int(0.7*len(train_images))
#Normalization of input pixel data
train_iter = mx.io.NDArrayIter(train_images[:ntrain, :]/255, train_labels[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(train_images[ntrain:, :]/255, train_labels[ntrain:], batch_size)
test_iter = mx.io.NDArrayIter(test_images/255, test_labels, batch_size)

train_loader = DataIterLoader(train_iter)
val_loader = DataIterLoader(val_iter)
test_loader = DataIterLoader(test_iter)


# In[7]:


net2 = nn.HybridSequential()

with net2.name_scope():
    net2.add(
        nn.Flatten(),
        nn.Dense(1024, activation='relu'),
        nn.Dense(512, activation='relu'),
        nn.Dense(256, activation='relu'),
        nn.Dense(10, activation=None)  # loss function includes softmax already, see below
    )
net2.hybridize()


# In[8]:


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


# In[9]:


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


# ## Getting internal features 

# In[25]:


model=mx.sym.load('vanilla_net2-symbol.json')
internals=model.get_internals()
ls=internals.list_outputs()
ls


# In[30]:


int1=internals[ls[4]]
sym1 = gluon.nn.SymbolBlock(outputs=int1, inputs=mx.sym.var('data'))
sym1.collect_params().load('vanilla_net2-0000.params', ctx=mx.cpu(0), ignore_extra=True)
hidden_training1=sym1(mx.nd.array(train_images)).asnumpy()
hidden_testing1=sym1(mx.nd.array(test_images)).asnumpy()


# In[32]:


int2=internals[ls[8]]
sym2 = gluon.nn.SymbolBlock(outputs=int2, inputs=mx.sym.var('data'))
sym2.collect_params().load('vanilla_net2-0000.params', ctx=mx.cpu(0), ignore_extra=True)
hidden_training2=sym2(mx.nd.array(train_images)).asnumpy()
hidden_testing2=sym2(mx.nd.array(test_images)).asnumpy()


# In[33]:


int3=internals[ls[12]]
sym3 = gluon.nn.SymbolBlock(outputs=int3, inputs=mx.sym.var('data'))
sym3.collect_params().load('vanilla_net2-0000.params', ctx=mx.cpu(0), ignore_extra=True)
hidden_training3=sym3(mx.nd.array(train_images)).asnumpy()
hidden_testing3=sym3(mx.nd.array(test_images)).asnumpy()


# ## Logistic Regression Classifier 

# In[34]:


from sklearn.linear_model import LogisticRegression
clf0 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(train_images, train_labels)
a0=clf0.score(train_images,train_labels)
b0=clf0.score(test_images,test_labels)
print('Accuracy with original data => training: {} and testing: {}'.format(a0,b0))


# In[35]:


clf1 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(hidden_training1, train_labels)
a1=clf1.score(hidden_training1,train_labels)
b1=clf1.score(hidden_testing1,test_labels)
print('Accuracy with 1st hidden layer data => training: {} and testing: {}'.format(a1,b1))


# In[36]:


clf2 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(hidden_training2, train_labels)
a2=clf2.score(hidden_training2,train_labels)
b2=clf2.score(hidden_testing2,test_labels)
print('Accuracy with 2nd hidden layer data => training: {} and testing: {}'.format(a2,b2))


# In[37]:


clf3 = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(hidden_training3, train_labels)
a3=clf3.score(hidden_training3,train_labels)
b3=clf3.score(hidden_testing3,test_labels)
print('Accuracy with 3rd hidden layer data => training: {} and testing: {}'.format(a3,b3))

