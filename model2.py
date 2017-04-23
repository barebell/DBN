import os
import timeit
import numpy as np
import math
import tensorflow as tf
import DBN_input

BATCH_SIZE = 64
NUM_CLASSES = 6
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 300
MAX_STEP = 500



class LogisticRegression(object):
    """Multi-class logistic regression class"""
    def __init__(self, inpt, n_in, n_out):
        """
        inpt: tf.Tensor, (one minibatch) [None, n_in]
        n_in: int, number of input units
        n_out: int, number of output units
        """
        # weight
        self.W = tf.Variable(tf.zeros([n_in, n_out], dtype=tf.float32))
        # bias
        self.b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32)
        # activation output
        self.output = tf.nn.softmax(tf.matmul(inpt, self.W) + self.b)
        # prediction
        self.y_pred = tf.argmax(self.output, axis=1)
        # keep track of variables
        self.params = [self.W, self.b]

    def cost(self, y):
        """
        y: tf.Tensor, the target of the input
        """
        # cross_entropy
        return -tf.reduce_mean(tf.reduce_sum(y * tf.log(self.output), axis=1))

    def accuarcy(self, y):
        """errors"""
        correct_pred = tf.equal(self.y_pred, tf.argmax(y, axis=1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class HiddenLayer(object):
    """Typical hidden layer of MLP"""
    def __init__(self, inpt, n_in, n_out, W=None, b=None,
                 activation=tf.nn.sigmoid):
        """
        inpt: tf.Tensor, shape [n_examples, n_in]
        n_in: int, the dimensionality of input
        n_out: int, number of hidden units
        W, b: tf.Tensor, weight and bias
        activation: tf.op, activation function
        """
        #if W is None:
        bound_val = 4.0*np.sqrt(6.0/(n_in + n_out))
        W = tf.Variable(tf.random_uniform([n_in, n_out], minval=-bound_val, maxval=bound_val),
                            dtype=tf.float32, name="W")
       # if b is None:
        b = tf.Variable(tf.zeros([n_out,]), dtype=tf.float32, name="b")

        self.W = W
        self.b = b
        # the output
        sum_W = tf.matmul(inpt, self.W) + self.b
        self.output = activation(sum_W) if activation is not None else sum_W
        # params
        self.params = [self.W, self.b]


class RBM(object):
    """A Restricted Boltzmann Machines class"""
    def __init__(self, inpt=None, n_visiable=24300, n_hidden=10000, W=None,
                 hbias=None, vbias=None):
        """
        :param inpt: Tensor, the input tensor [None, n_visiable]
        :param n_visiable: int, number of visiable units
        :param n_hidden: int, number of hidden units
        :param W, hbias, vbias: Tensor, the parameters of RBM (tf.Variable)
        """
        self.n_visiable = n_visiable
        self.n_hidden = n_hidden
        """input need to be config wether given or not """
        # Optionally initialize input
        if inpt is None:
            inpt = tf.placeholder(dtype=tf.float32, shape=[None, self.n_visiable])
        self.input = inpt
        # Initialize the parameters if not given
        if W is None:
            bounds = -4.0 * np.sqrt(6.0 / (self.n_visiable + self.n_hidden))
            W = tf.Variable(tf.random_uniform([self.n_visiable, self.n_hidden], minval=-bounds,
                                              maxval=bounds), dtype=tf.float32)
        if hbias is None:
            hbias = tf.Variable(tf.zeros([self.n_hidden,]), dtype=tf.float32)
        if vbias is None:
            vbias = tf.Variable(tf.zeros([self.n_visiable,]), dtype=tf.float32)
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        # keep track of parameters for training (DBN)
        self.params = [self.W, self.hbias, self.vbias]
    
    def propup(self, v):
        """Compute the sigmoid activation for hidden units given visible units"""
        return tf.nn.sigmoid(tf.matmul(v, self.W) + self.hbias)

    def propdown(self, h):
        """Compute the sigmoid activation for visible units given hidden units"""
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.vbias)
    
    def sample_prob(self, prob):
        """Do sampling with the given probability (you can use binomial in Theano)"""
        return tf.nn.relu(tf.sign(prob - tf.random_uniform(tf.shape(prob))))
    
    def sample_h_given_v(self, v0_sample):
        """Sampling the hidden units given visiable sample"""
        h1_mean = self.propup(v0_sample)
        h1_sample = self.sample_prob(h1_mean)
        return (h1_mean, h1_sample)
    
    def sample_v_given_h(self, h0_sample):
        """Sampling the visiable units given hidden sample"""
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.sample_prob(v1_mean)
        return (v1_mean, v1_sample)
    
    def gibbs_vhv(self, v0_sample):
        """Implement one step of Gibbs sampling from the visiable state"""
        h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return (h1_mean, h1_sample, v1_mean, v1_sample)

    def gibbs_hvh(self, h0_sample):
        """Implement one step of Gibbs sampling from the hidden state"""
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return (v1_mean, v1_sample, h1_mean, h1_sample)

    def free_energy(self, v_sample):
        """Compute the free energy"""
        wx_b = tf.matmul(v_sample, self.W) + self.hbias
        vbias_term = tf.matmul(v_sample, tf.expand_dims(self.vbias, axis=1))
        hidden_term = tf.reduce_sum(tf.log(1.0 + tf.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def get_train_ops(self, learning_rate=0.1, k=1, persistent=None):
        """
        Get the training opts by CD-k
        :params learning_rate: float
        :params k: int, the number of Gibbs step (Note k=1 has been shown work surprisingly well)
        :params persistent: Tensor, PCD-k (TO DO:)
        """
        # Compute the positive phase
        ph_mean, ph_sample = self.sample_h_given_v(self.input)
        # The old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent

        # Use tf.while_loop to do the CD-k
        cond = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: i < k
        body = lambda i, nv_mean, nv_sample, nh_mean, nh_sample: (i+1, ) + self.gibbs_hvh(nh_sample)
        i, nv_mean, nv_sample, nh_mean, nh_sample = tf.while_loop(cond, body, loop_vars=[tf.constant(0), tf.zeros(tf.shape(self.input)), 
                                                            tf.zeros(tf.shape(self.input)), tf.zeros(tf.shape(chain_start)), chain_start])

        # Compute the update values for each parameter
        update_W = self.W + learning_rate * (tf.matmul(tf.transpose(self.input), ph_mean) - 
                                tf.matmul(tf.transpose(nv_sample), nh_mean)) / tf.to_float(tf.shape(self.input)[0])  # use probability
        update_vbias = self.vbias + learning_rate * (tf.reduce_mean(self.input - nv_sample, axis=0))   # use binary value
        update_hbias = self.hbias + learning_rate * (tf.reduce_mean(ph_mean - nh_mean, axis=0))       # use probability
        # Assign the parameters new values
        new_W = tf.assign(self.W, update_W)
        new_vbias = tf.assign(self.vbias, update_vbias)
        new_hbias = tf.assign(self.hbias, update_hbias)
        #tf.summary.scalar('w'，new_W)
        chain_end = tf.stop_gradient(nv_sample)   # do not compute the gradients
        cost = tf.reduce_mean(self.free_energy(self.input)) - tf.reduce_mean(self.free_energy(chain_end))
        # Compute the gradients
        gparams = tf.gradients(ys=[cost], xs=self.params)
        new_params = []
        for gparam, param in zip(gparams, self.params):
            new_params.append(tf.assign(param, param - gparam*learning_rate))

        if persistent is not None:
            new_persistent = [tf.assign(persistent, nh_sample)]
        else:
            new_persistent = []
        return new_params + new_persistent  # use for training

    def get_reconstruction_cost(self):
        """Compute the cross-entropy of the original input and the reconstruction"""
        activation_h = self.propup(self.input)
        activation_v = self.propdown(activation_h)
        # Do this to not get Nan
        activation_v_clip = tf.clip_by_value(activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        reduce_activation_v_clip = tf.clip_by_value(1.0 - activation_v, clip_value_min=1e-30, clip_value_max=1.0)
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.input*(tf.log(activation_v_clip)) + 
                                    (1.0 - self.input)*(tf.log(reduce_activation_v_clip)), axis=1))
        return cross_entropy    
    def reconstruct(self, v):
        """Reconstruct the original input by RBM"""
        h = self.propup(v)
        return self.propdown(h) 

class DBN(object):
    """
    An implement of deep belief network
    The hidden layers are firstly pretrained by RBM, then DBN is treated as a normal
    MLP by adding a output layer.
    """
    def __init__(self, x, y, n_in=784, n_out=10, hidden_layers_sizes=[500, 500,500]):
        """
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the hidden layer sizes
        """
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # normal sigmoid layer
        self.rbm_layers = []   # RBM layer
        self.params = []       # keep track of params for training

        # Define the input and output
        self.x = x
        self.y = y

        # Contruct the layers of DBN
        for i in range(self.n_layers):
            if i == 0:
                layer_input = self.x
                input_size = n_in
            else:
                layer_input = self.layers[i-1].output
                input_size = hidden_layers_sizes[i-1]
            # Sigmoid layer
            sigmoid_layer = HiddenLayer(inpt=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                    activation=tf.nn.sigmoid)
            self.layers.append(sigmoid_layer)
            # Add the parameters for finetuning
            self.params.extend(sigmoid_layer.params)
            # Create the RBM layer
            self.rbm_layers.append(RBM(inpt=layer_input, n_visiable=input_size, n_hidden=hidden_layers_sizes[i],
                                        W=sigmoid_layer.W, hbias=sigmoid_layer.b))
        # We use the LogisticRegression layer as the output layer
        self.output_layer = LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1],
                                                n_out=n_out)
        self.params.extend(self.output_layer.params)
        # The finetuning cost
        self.cost = self.output_layer.cost(self.y)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
    
    def pretrain(self, sess, batch_size=50, pretraining_epochs=10, lr=0.1, k=1, 
                    display_step=1):
        """
        Pretrain the layers (just train the RBM layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modidy this function if you do not use the desgined mnist)
        :param batch_size: int
        :param lr: float
        :param k: int, use CD-k
        :param pretraining_epoch: int
        :param display_step: int
        """
        print('Starting pretraining...\n')
        start_time = timeit.default_timer()
        # Pretrain layer by layer
        for i in range(self.n_layers):
            cost = self.rbm_layers[i].get_reconstruction_cost()
            train_ops = self.rbm_layers[i].get_train_ops(learning_rate=lr, k=k, persistent=None)
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                #x_batch = sess.run(X_train)
                    # 训练
                sess.run(train_ops)
                    # 计算cost
                avg_cost += sess.run(cost)
                # 输出
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost/1000))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, trainSet, training_epochs=10, batch_size=100, lr=0.1,
                   display_step=1):
        """
        Finetuing the network
        """
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(
            self.cost, var_list=self.params)
        for epoch in range(training_epochs):
            avg_cost = 0.0
            batch_num = int(trainSet.train.num_examples / batch_size)
            for i in range(batch_num):
                x_batch, y_batch = trainSet.train.next_batch(batch_size)
                # 训练
                sess.run(train_op, feed_dict={self.x: x_batch, self.y: y_batch})
                # 计算cost
                avg_cost += sess.run(self.cost, feed_dict=
                {self.x: x_batch, self.y: y_batch}) / batch_num
            # 输出
            if epoch % display_step == 0:
                val_acc = sess.run(self.accuracy, feed_dict={self.x: trainSet.validation.images,
                                                       self.y: trainSet.validation.labels})
                print("\tEpoch {0} cost: {1}, validation accuacy: {2}".format(epoch, avg_cost, val_acc))

        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))


if __name__ == "__main__":
    """
    images_batchs : [64,90*90*3]
    label_batchs: [64]
    images: [100,90,90,3]
    labels: [100]
    only batch_size has changed
    """
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    tf.train.start_queue_runners(sess=sess)
    images_batchs,label_batchs= DBN_input.distorted_inputs(BATCH_SIZE)
    #images_batchs = tf.as_dtype(np.float32)
    print(images_batchs.shape)
    #print(label_batchs)
    #print(sess.run(label_batchs))
    images,labels= DBN_input.inputs(batch_size = 100)
    
    
    dbn = DBN(x= images_batchs ,y = label_batchs, n_in=24300, n_out=6, hidden_layers_sizes=[8000, 4000,1000])

    
    # set random_seed
    tf.set_random_seed(seed=1111)
    dbn.pretrain(sess)
    #dbn.finetuning(sess, images = images,labels = labels)
