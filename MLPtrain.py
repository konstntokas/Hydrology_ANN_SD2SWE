#inputs: x_train: input for training 
#        y_train: target for training 
#        x_val: input for MLP validation
#        y_val: target for MLP validation
#        nb_epochs: number of epochs in training 
#        nb_member: number of members in ensemble 
#        nb_hid: number of neurons in hidden layer 
#        save_path: path to save the neural networks for later use 
#        activ_fc: activation function in hidden layer
#                  'tanh', 'ReLU', 'LeakyReLU' are possible 
#        init_w: float value for weights initialisation with U(-init_w, init_w)
#        init_b: float value for weights initialisation with U(-init_b, init_b)
#        potAlg: determine optimisation Algorithm: 'Adadelta' and 'RMSProp' are possible
#        batch_size: number of records in one minibatch 
#        shuf_data: 0 = no shuffling of the data between the epochs
#                   1 = shuffling of the data between the epochs
#    
#outputs: final: results of network in ensemble by using the x_test data as input
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

from datetime import datetime 
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle 


def MLP(x_train, y_train, nb_epochs, nb_members, nb_hid, save_path, 
        activ_fc, init_w, init_b, optAlg, batch_size, shuf_data):
    
    # standardize data 
    # create scaler
    scalerIn = StandardScaler()
    scalerOut = StandardScaler()
    # fit scaler on dataIn and target and transform it
    scalerIn.fit(x_train)
    x_train_std = scalerIn.transform(x_train)
    scalerOut.fit(y_train.reshape(-1, 1))
    y_train_std = scalerOut.transform(y_train.reshape(-1, 1))     
    
    # Optimisation Parameters
    if optAlg == 'Adadelta':
        learning_rate = 1
    elif optAlg == 'RMSProp':
        learning_rate = 0.00001
    decay=0.9
    momentum=0.0
    epsilon=1e-8
    display_step = 1
    # if batch size None, then use entrie batch 
    if batch_size == None: 
        batch_size = len(y_train_std)
    
    # Network Parameters
    nb_input = x_train.shape[1]
    nb_output = 1 
    
    # tf Graph input
    X = tf.placeholder("float", [None, nb_input],name="input")
    Y = tf.placeholder("float", [None, nb_output])
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_uniform([nb_input, nb_hid], minval=-init_w, maxval=init_w), name='Whid'),
        'out': tf.Variable(tf.random_uniform([nb_hid, nb_output], minval=-init_w, maxval=init_w), name='Wout')
    }
    biases = {
        'b1': tf.Variable(tf.random_uniform([nb_hid], minval=-init_b, maxval=init_b), name='Bhid'),
        'out': tf.Variable(tf.random_uniform([nb_output], minval=-init_b, maxval=init_b), name='Bout')
    }
    
    
    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        if activ_fc == 'tanh':
            layer_1 = tf.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        elif activ_fc == 'ReLU':
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        elif activ_fc == 'Leaky_ReLU':
            layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']), alpha=0.1)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'], name="op_to_restore")
        return out_layer
    
    # Construct model
    output = multilayer_perceptron(X)
    
    # Original loss function
    loss_op = tf.reduce_mean(tf.squared_difference(output, Y))
    if optAlg == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=decay, epsilon=epsilon)
    elif optAlg == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=decay, momentum=momentum, epsilon=epsilon)
    train_op = optimizer.minimize(loss_op)
    
    #Create a saver object which will save all the variables
    saver = tf.train.Saver(max_to_keep=nb_members)
    
      
    for mb in range(nb_members):
        print('member {}'.format(mb))
        print(datetime.now())
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
        
            # Training cycle
            for epoch in range(nb_epochs):
                # shuffle data if required
                if shuf_data == 1: 
                    p = np.random.permutation(len(x_train_std))
                    x_train_pt = x_train_std[p]
                    y_train_pt = y_train_std[p]
                

                total_batch = int(len(y_train_std)/batch_size)
                # Loop over all minibatches
                for i in range(total_batch):
                    batch_x = x_train_pt[i*batch_size:(i+1)*batch_size, :]
                    batch_y = y_train_pt[i*batch_size:(i+1)*batch_size]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c_train = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                    
                # calculate the loss on train set 
                y_predict_std = sess.run(output, feed_dict={X: x_train_std}).flatten()
                y_predict = scalerOut.inverse_transform(y_predict_std)   
                c_train = np.mean((y_predict - y_train)**2)

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost train={:.2f}".format(c_train))
            print("Optimization Finished!")
        
            
            # save trained model
            saver.save(sess, save_path + '/mb_{}.ckpt'.format(mb))
             
            sess.close()
            
    # save scaler         
    pickle.dump([scalerIn, scalerOut], open(save_path + '/scaler', "wb"))
    
    print('training of ensemble finished')

