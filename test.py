import tensorflow as tf
import numpy as np
import matplotlib as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) # Weight 矩阵就是in_size * out_size, input 乘以它，就成了output
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # Biases vector不为零
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs # inputs 和 outputs都是tensor 向量

# Build Neural Network

# Make up some real data
x_data = np.linspace(-1,1,300)[:, np.newaxis] # 把x_data构建成column vector
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu) # placeholder就是一个变量指针，在session run中用dict进行映射。所以可以直接作为传入函数。
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real dataEllipsis
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# # plot the real data
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)
# plt.ion()
# plt.show()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to visualize the result and improvement
        # try:
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
        print(sess.run(prediction, feed_dict={xs: x_data}))

        # plot the prediction
        #lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        #plt.pause(1)