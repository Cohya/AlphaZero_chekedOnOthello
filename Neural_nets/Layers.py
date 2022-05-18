
import tensorflow as tf 
import numpy as np 

decay = 0.99
class Dense():
    def __init__(self, M1, M2, layer_Num ,apply_batch_norm = False, apply_dropout = False, drop_rate = 0, f = tf.nn.relu, use_bias = True):
        W0 = np.random.randn(M1, M2).astype(np.float32) * np.sqrt(2./float(M1))
        self.W = tf.Variable(initial_value = W0, name = 'W_dense_%i' % layer_Num)
        
        self.use_bias = use_bias
        
        if self.use_bias:
            self.b  = tf.Variable(initial_value = tf.zeros(shape = [M2,]), name = 'b_%i' % layer_Num)
        
        self.apply_batch_norm = apply_batch_norm
        self.apply_dropout = apply_dropout
        self.drop_rate = drop_rate
        
        if self.apply_batch_norm:
            self.gamma = tf.Variable(initial_value = tf.ones(shape = [M2,]), name = "gamma_%i" % layer_Num)
            self.beta = tf.Variable(initial_value = tf.zeros(shape = [M2,]), name = "beta_%i" % layer_Num)
            
            self.running_mean = tf.Variable(initial_value = tf.zeros(shape = [M2,]), name = "running_mean_%i" % layer_Num,
                                            trainable = False) # not trainable 
            
            self.running_var = tf.Variable(initial_value = tf.ones(shape = [M2,]) ,
                                           name = "running_var_%i" % layer_Num, 
                                           trainable = False) # trainable = False
            
            self.nurmalization_params = [self.running_mean, self.running_var]
            
        self.f = f
        self.id = layer_Num
        self.trainable_params = [self.W]
        if self.use_bias:
            self.trainable_params += [self.b]
        
        if self.apply_batch_norm:
            self.trainable_params += [self.gamma, self.beta]
            
    def forward(self, X, is_training):
        Z = tf.matmul(X,self.W) 
        if self.use_bias:
            Z +=  self.b
        
        if self.apply_batch_norm:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(Z, [0])
                
                self.running_mean.assign(self.running_mean * decay + batch_mean * (1 - decay))
               
                self.running_var.assign(self.running_var * decay + batch_var * (1 - decay))
                self.normalization_params = [self.running_mean, self.running_var]
                # with tf.control_dependencies([batch_mean, batch_var]):
                Z = tf.nn.batch_normalization(Z ,
                                                batch_mean,
                                                batch_var,
                                                self.beta,
                                                self.gamma,
                                                1e-3)
            else:
                
                  # this is for the testing 
                Z = tf.nn.batch_normalization(Z,
                                            self.running_mean,
                                            self.running_var,
                                            self.beta,
                                            self.gamma,
                                            1e-3)  
        if self.apply_dropout:
            if is_training:
                Z = tf.nn.dropout(Z, self.drop_rate)
                
        return self.f(Z)
        
class ConvLayer(object):
    def __init__(self, mi, mo, dim_output_x, dim_output_y, layer_Num, filtersz = 4, stride = 1,  f=tf.nn.relu, Pad = 'SAME',
                 apply_zero_padding = False, apply_dropout = False, apply_batch_norm =False, drop_rate = 0,add_bias = False):
        # mi = input feature map size 
        # mo = output feature map size
        # Gets an existing variable with this name or creat a new one 
        
        # if apply_zero_padding: # for applying batch norm befiore zero padding
        #     dim_output_x = dim_output_x -2
        #     dim_output_y = dim_output_y -2
        name = str(layer_Num)
        self.add_bias = add_bias
        self.W = tf.Variable(initial_value = tf.random.normal(shape = [filtersz,filtersz, mi, mo],
                                                              stddev = 0.02), name = "W_conv_%s" % name) 
                              
        if self.add_bias:
            self.b = tf.Variable(initial_value = tf.zeros(mo,), name = "b_%s" % name)
         
        self.apply_batch_norm = apply_batch_norm                       
        
        #for batch norm 
        if self.apply_batch_norm:
            self.gamma = tf.Variable(initial_value = tf.ones(shape = [mo,]) , name ="gamma_conv_%s" % name)
                                         
            self.beta = tf.Variable(initial_value  = tf.zeros(shape = [mo,]), name = "beta_conv_%s" % name)
            
            self.running_mean = tf.Variable(initial_value = tf.zeros(shape = [dim_output_x,dim_output_y, mo]) ,
                                                name = "running_mean_conv_%s" % name,
                                                trainable = False) # Trainable = False [dim_output_x,dim_output_y, mo]
            
            self.running_var = tf.Variable(initial_value = tf.ones(shape = [dim_output_x,dim_output_y, mo]) ,
                                           name = "running_var_conv_%s" % name,
                                           trainable = False) # trainable = False [dim_output_x,dim_output_y, mo]
            
            self.normalization_params = [self.running_mean, self.running_var]
            
        self.name = name 
        self.f = f
        self.stride = stride
        self.pad = Pad
        self.drop_rate = drop_rate
        self.apply_dropout = apply_dropout
        self.apply_zero_padding = apply_zero_padding
        
        if self.add_bias:
            self.trainable_params = [self.W, self.b]
        else:
            self.trainable_params  = [self.W]
        
        if self.apply_batch_norm:
            self.trainable_params  += [self.gamma, self.beta]
        
    def forward(self, X, is_training):

        conv_out = tf.nn.conv2d(X, filters = self.W, strides=[1, self.stride, self.stride,1],
                                padding=self.pad )

        if self.add_bias:
            conv_out = tf.nn.bias_add(conv_out, self.b)
        # print(conv_out.shape)
        
        # apply batch norm 
        
        if self.apply_batch_norm:
            if is_training:
                batch_mean, batch_var = tf.nn.moments(conv_out, [0,1,2]) # this is batch on axes = 3
                
                self.running_mean.assign(self.running_mean * decay + batch_mean * (1 - decay))
               
                self.running_var.assign(self.running_var * decay + batch_var * (1 - decay))
                    
                self.normalization_params = [self.running_mean, self.running_var]   
                # with tf.control_dependencies([batch_mean, batch_var]):
                conv_out = tf.nn.batch_normalization(conv_out ,
                                                 batch_mean, 
                                                 batch_var ,
                                                self.beta,
                                                self.gamma,
                                                1e-3)
            else:
                
                  # this is for the testing 
                conv_out = tf.nn.batch_normalization(conv_out,
                                            self.running_mean,
                                            self.running_var,
                                            self.beta,
                                            self.gamma,
                                            1e-3)
                
        if self.apply_dropout:
            if is_training:
                conv_out = tf.nn.dropout(conv_out, self.drop_rate)
            # print("Drop")
            
            
        conv_out = self.f(conv_out)
        
        if self.apply_zero_padding:
            conv_out = tf.keras.layers.ZeroPadding2D(padding=1)(conv_out)
                
       
        return conv_out 