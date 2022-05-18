import tensorflow as tf
import numpy as np 
from utiles import dotdict
from Layers import Dense, ConvLayer
from sklearn.utils import shuffle 
import pickle 
import os 

        
class OthelloNNet(object):
    def __init__(self, game, args):
        
        #Game parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        
        # The net 
        # Conve layers
        # mi, mo, apply_batch_norm, dim_output_x, dim_output_y,layer_Num, filtersz = 4, stride = 1,  f=tf.nn.relu, Pad = 'SAME',
        #          apply_zero_padding = 'False', apply_dropout = False, drop_rate = 0,
        #add_bias = False
        # (mo, kernel_size, padding, use_bias, activation, apply_batch_norm ) <-- for conv layer 
        # (M1, M2, apply_batch_norm, apply_dropout, layer_Num , drop_rate = 0, f = tf.nn.relu)
        # (M2, apply_batch_norm, apply_dropout, drop_rate, activation, use_bias) <-- FOR DENSE LAYER 
        
        nn_details = dotdict({'conv_layers': [(args.num_channels,3, 'SAME', False, tf.nn.relu, True),
                                      (args.num_channels,3, 'SAME', False, tf.nn.relu, True),
                                      (args.num_channels,3, 'VALID', False, tf.nn.relu, True),
                                      (args.num_channels,3, 'VALID', False, tf.nn.relu, True)],
                      'denseLayer': [(1024,True, True, args.dropout, tf.nn.relu, False ),
                                     (512,True, True, args.dropout, tf.nn.relu, False )],
                      'policy_layer': [self.action_size, False, False, 0, tf.nn.softmax, True],
                      'value_layer': [1, False, False, 0, tf.nn.tanh, True]})
        
        # Let's build the net 
        
        self.convLayers = []
        
        mi = 1 # the third dimention of the boardgam 
        dim_output_x = self.board_x
        dim_output_y = self.board_y
        count = 0 
        for mo, kernel_size,padding, add_bias, activation, apply_BN in nn_details.conv_layers:
            
            if padding == 'VALID':
                dim_output_x = dim_output_x - 2
                dim_output_y = dim_output_y - 2
              
           
            conv = ConvLayer(mi = mi, mo = mo,
                            dim_output_x = dim_output_x, dim_output_y = dim_output_y,
                            layer_Num= count, 
                            filtersz = kernel_size,
                            f = activation,
                            apply_batch_norm= apply_BN,
                            add_bias= add_bias,
                            Pad = padding)

            self.convLayers.append(conv)
            mi = mo
            
            count += 1
            
        
        ## Let's collect the dense layers
        M1 = dim_output_x  * dim_output_y * mi 
        self.M1 = M1
        self.denseLayers = []
        
        for M2, apply_batch_norm, apply_dropout, drop_rate, activation, use_bias in nn_details.denseLayer:
            layer = Dense(M1 = M1, M2 = M2,
                          layer_Num=count, 
                          apply_batch_norm= apply_batch_norm,
                          apply_dropout= apply_dropout,
                          drop_rate = drop_rate,
                          f = activation,
                          use_bias= use_bias)
            
            M1 = M2
            count += 1
            
            self.denseLayers.append(layer)
            
        
        # Now let's define the policy layer 
        M2, apply_batch_norm, apply_dropout, drop_rate, activation, use_bias = nn_details.policy_layer
        self.policyLayer = Dense(M1 = M1, M2 = M2, 
                                  layer_Num=count,
                                  apply_batch_norm = apply_batch_norm,
                                  apply_dropout= apply_dropout,
                                  drop_rate = drop_rate,
                                  f = activation,
                                  use_bias = use_bias)
         
        # Next the value function estimation
        M2, apply_batch_norm, apply_dropout, drop_rate, activation, use_bias = nn_details.value_layer
        self.valueLayer = Dense(M1 = M1, M2 = M2, 
                                  layer_Num=count,
                                  apply_batch_norm = apply_batch_norm,
                                  apply_dropout= apply_dropout,
                                  drop_rate = drop_rate,
                                  f = activation,
                                  use_bias = use_bias)
        
        
        ## let's collect the trainable params 
        self.trainable_params = []
        
        for layer in self.convLayers:
            self.trainable_params += layer.trainable_params
            
        for layer in self.denseLayers:
            self.trainable_params += layer.trainable_params
            
        self.trainable_params += self.policyLayer.trainable_params
        self.trainable_params += self.valueLayer.trainable_params
        
        self.opt  = tf.keras.optimizers.Adam(lr = args.lr)
        
        
    def forward(self, X, is_training):
        z = X

        if len(z.shape) !=4:
            z = np.expand_dims(z, axis =-1)
        assert len(z.shape) == 4, "Make sure that your input has 4 dims (batch_size, h,w,c)"
        
        # convolution layers
        
        for conv in self.convLayers:
            z = conv.forward(z, is_training = is_training)
        
        
        ## flatten procedure before dense layer 
        n ,h,w,c = z.shape 
        z = tf.reshape(z , shape = (n, h*w*c))
        # print("shape:", z.shape)
        # dense layers
        
        for layer in self.denseLayers:
            z = layer.forward(z, is_training = is_training)
            # print(z.shape)
        
        ## Now we need to split into two heads (policy and value)
        
        policy  = self.policyLayer.forward(z,is_training = is_training)
        value = self.valueLayer.forward(z, is_training = is_training)
        
        return policy, value
    
    def cost(self, X, target_policy, target_value, is_training = False):
        
        policy_hat, value_hat = self.forward(X, is_training = is_training)
        
        # minimize cross-entropy is maximizing p(theta) be closer to Pi with respect to theta, maximizing the 
        # the log liklihood.maximizing the likelihood with respect to the parameters {theta }
        #is the same as minimizing the cross-entropy.
        cross_entropy = tf.reduce_mean(- tf.reduce_sum(target_policy * tf.math.log(policy_hat), axis = 1))
        mse = tf.reduce_mean(tf.math.squared_difference(target_value,value_hat))
        cost_i = cross_entropy + mse
        
        return cost_i
        
    def optimization_step(self,Xbatch, target_policyBatch, target_valueBatch):
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            cost_i = self.cost(Xbatch, target_policyBatch, target_valueBatch, is_training = True)
            
            
        gradients = tape.gradient(cost_i, self.trainable_params)
        self.opt.apply_gradients(zip(gradients, self.trainable_params))
        
        return cost_i
        
    def fit(self, X, Y,  batch_sz, epochs):
        
        target_policy, target_value = Y
        n_batchs = len(X) // batch_sz
        
        for epoch in range(epochs):
            X , target_policy, target_value = shuffle(X,target_policy, target_value)
            
            for j in range(n_batchs):
                Xbatch  = X[j*batch_sz : (j + 1) * batch_sz, ...]
                target_policyBatch = target_policy[j*batch_sz:(j+1)*batch_sz,...]
                target_valueBatch = target_value[j*batch_sz: (j+1)*batch_sz,...]
                
                cost_i = self.optimization_step(Xbatch, target_policyBatch, target_valueBatch)
                
            
            if epoch % 2 ==0 :
                print("Epoch: ", epoch, "Cost: ", cost_i.numpy())
                
        
        return 
        
      
    def predict(self, X, is_training = False):
        policy, value = self.forward(X, is_training)
        return policy, value 
    
        
        
        
    def save_weights(self, folder = 'chekpoint', file_name = "checkpoint"):
         self.checkFolders(folder)
         file_name  = os.path.join(folder, file_name)
         
         with open(file_name, 'wb') as file :
             pickle.dump(self.trainable_params, file)
         
        
         print("Weights checkPoint was saved!")
             
             
    def load_weights(self, folder, file_name):
        file_path = os.path.join(folder, file_name)
        
        try:
            with open(file_path, 'rb') as file:
                trainable_paramsLoad = pickle.load(file)
                for tf_param, param in zip(self.trainable_params, trainable_paramsLoad):
                    tf_param.assign(param)
                    
                
                print("Weights were load successfully!!")
                
        except OSError:
               print("File %s doesn't exist!" % file_path)


    def checkFolders(self,folder):
        print(folder)
        if not os.path.isdir(folder):
            path = os.getcwd() + folder
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of the directory %s failed " % path)
            else:
                print("Successful creation of the following path for weights %s" % path)
                
        
                    
            
           
        
        
# game =0 
# net = OthelloNNet(game, args)      

# x = tf.random.normal(shape = (10,10,10,1))

# p,v = net.forward(x, False )


# z =  x
# for layer in net.convLayers:
#     print("z", z.shape)
#     z = layer.forward(z, is_training = False)
#     print("z", z.shape)
# a1 = tf.zeros(shape = (1,3,3,6)) + 1.
# a2 = tf.zeros(shape = (1,3,3,6)) + 2
# a3 = tf.zeros(shape = (1,3,3,6)) + 3.
# a = tf.concat([a1,a2,a3], axis = 0)
# layer = tf.keras.layers.BatchNormalization(axis = [3])      
# b = layer(a)

# batch_mean,batch_var= tf.nn.moments(a, axes = [0,1,2])
# running_var = tf.Variable(tf.ones(shape = a.shape[1:]))
# running_mean  = tf.Variable(tf.zeros(shape = a.shape[1:]))
# decay = 0.99999999
# running_mean.assign(running_mean * decay + batch_mean * (1 - decay))
   
# running_var.assign(running_var * decay + batch_var * (1 - decay))
# b2 = tf.nn.batch_normalization(a, running_mean
#                                ,running_var,0,1,1e-3)
  
# print(np.sum(b2 - b) )   

# from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, Dropout
# layer =  Dense(5)
# layer2 = BatchNormalization(axis=1)
# layer3 = Activation('relu')
# layer4 = Dropout(0.5)

# X = tf.random.normal(shape = (3,10))

# y1 = layer(X)
# y2 = layer2(y1)
# y3 = layer3(y2, training = True)
# y4 = layer4(y3)

# print("X:", X)
# print("y1:", y1)
# print("y2:", y2)
# print("y3:", y3)
# print("y4:", y4)
