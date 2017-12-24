
import mnist_reader as reader
import numpy as np
import tensorflow as tf
from scipy import misc
import IPython

def Generator(inputs,training=True):
    with tf.variable_scope('generator') as scope:
        if not training:
            scope.reuse_variables()
        with tf.variable_scope('dense1'):
            dconv = tf.layers.dense(inputs,7*7*16,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            dconv = tf.reshape(dconv,shape=[-1,7,7,16])
        with tf.variable_scope('deconv1'):
            dconv = tf.layers.conv2d_transpose(dconv,32,kernel_size=3,strides=(1, 1),padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            dconv = tf.layers.batch_normalization(dconv,name='bn_dconv1',training=training)
            dconv = tf.nn.relu(dconv)
        with tf.variable_scope('deconv2'):
            dconv = tf.layers.conv2d_transpose(dconv,16,kernel_size=5,strides=(2, 2),padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            dconv = tf.layers.batch_normalization(dconv,name='bn_dconv2',training=training)
            dconv = tf.nn.relu(dconv)
        with tf.variable_scope('deconv21'):
            dconv = tf.layers.conv2d_transpose(dconv,1,kernel_size=5,strides=(2, 2),padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            dconv = tf.layers.batch_normalization(dconv,name='bn_dconv3',training=training)
            dconv = tf.nn.sigmoid(dconv)
        return dconv

def Discriminator(inputs,reuse=False):
    # print inputs
    with tf.variable_scope('discriminator',reuse=reuse):
        with tf.variable_scope('conv1'):
            conv = tf.layers.conv2d(inputs,filters=8,kernel_size=3,strides=(2, 2),padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.layers.batch_normalization(conv,name='bn_conv1',training=True)
            conv = tf.nn.relu(conv)
        with tf.variable_scope('conv2'):
            conv = tf.layers.conv2d(conv,filters=16,kernel_size=3,strides=(2, 2),padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.layers.batch_normalization(conv,name='bn_conv2',training=True)
            conv = tf.nn.relu(conv)
        with tf.variable_scope('conv3'):
            conv = tf.layers.conv2d(conv,filters=32,kernel_size=3,strides=(1, 1),padding='same',kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv = tf.layers.batch_normalization(conv,name='bn_conv3',training=True)
            conv = tf.nn.relu(conv)
        with tf.variable_scope('fc1'):
            shape = conv.get_shape().as_list()
            conv = tf.reshape(conv,shape=[-1,np.prod(shape[1:])])
            logit = tf.layers.dense(conv,1,activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        # shape = logit.get_shape().as_list()
    return logit
datasets = reader.load_mnist(train_dir='data',reshape=False)
train = datasets.train
# mask = train.labels == 5
# samples = train.images[mask]
samples = train.images[...]
data_size = samples.shape[0]

feature_length = 64

input_noise = tf.placeholder(shape=[None,feature_length],dtype=tf.float32)

generated = Generator(input_noise)
sampled = Generator(input_noise,training=False)

input_data = tf.placeholder(shape=[None,28,28,1],dtype=tf.float32)

logit_real = Discriminator(input_data,reuse=False)

logit_fake = Discriminator(generated,reuse=True)

loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_real,labels=tf.ones_like(logit_real))
loss_real = tf.reduce_mean(loss_real)

loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake,labels=tf.zeros_like(logit_fake))
loss_fake = tf.reduce_mean(loss_fake)

loss_discriminator = loss_real + loss_fake

loss_generater = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_fake,labels=tf.ones_like(logit_fake))

loss_generater = tf.reduce_mean(loss_generater)

trainables = tf.trainable_variables()
trainables_discriminator = [var for var in trainables if var.name.startswith('discriminator')]
trainables_generator = [var for var in trainables if var.name.startswith('generator')]

updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(updates):
    trainer = tf.train.MomentumOptimizer(learning_rate=0.0002,momentum=0.9)
    trainer_g = tf.train.MomentumOptimizer(learning_rate=0.0002,momentum=0.9)
    minimizer_discrimitor = trainer.minimize(loss_discriminator,var_list=trainables_discriminator)
    minimizer_generator = trainer_g.minimize(loss_generater,var_list=trainables_generator)
    gradient_fake = trainer.compute_gradients(loss_generater)
batch = 20
epoch = 20

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        for j in range(data_size//batch):
            train_data = samples[j*batch:j*batch+batch]

            noise = np.random.uniform(low=-1.0,high=1.0,size=[batch,feature_length])
            _,l_d = sess.run([minimizer_discrimitor,loss_discriminator],feed_dict={input_data:train_data,input_noise:noise})
            for k in range(2):
                _,l_g,g = sess.run([minimizer_generator,loss_generater,generated],feed_dict={input_data:train_data,input_noise:noise})
                noise = np.random.uniform(low=-1.0,high=1.0,size=[batch,feature_length])
            if(j!= 0 and j%20==0):
                print 'loss is ',l_d,l_g
        np.random.shuffle(samples)
        noises = np.random.uniform(low=-1.0,high=1.0,size=[data_size,feature_length])

    generate_size = 5
    noise = np.random.uniform(low=-1.0,high=1.0,size=[generate_size,feature_length])
    fakes = sess.run(sampled,feed_dict={input_noise:noise})


    fakes.reshape(generate_size,28,28)
    for i in range(generate_size):
        fake = fakes[i]
        fake = fake*255
        fake = fake.astype(np.uint8)
        misc.imsave('generated_%d.png'%i,fake.reshape(28,28))

    # IPython.embed()
