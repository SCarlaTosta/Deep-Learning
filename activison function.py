import tensorflow as tf



x = tf.constant([[1],
                 [2],
                 [3],
                 [4],
                 [5],
                 [6],
                 [7],
                 [8],
                 [9],
                 [10],
                 [11],
                 [12],
                 [13],
                 [14],
                 [15],
                 [16],],dtype= tf.float32)


y = x ** 2

x_train = x / 16.0
y_train = y / 256.0

tf.random.set_seed(12)

w1 = tf.Variable(tf.random.normal((1,32),stddev=0.1))
b1 = tf.Variable(tf.zeros((32,)))

w2 = tf.Variable(tf.random.normal((32,32),stddev=0.1))
b2 = tf.Variable(tf.zeros((32,)))

w3 = tf.Variable(tf.random.normal((32,1),stddev=0.1))
b3 = tf.Variable(tf.zeros((1,)))

learning_rat = 0.001

for epoc in range(100000):
    with tf.GradientTape() as Tape:
        z1 = tf.matmul(x_train,w1) + b1
        a1 = tf.nn.relu(z1)
        z2 = tf.matmul(a1,w2) + b2
        a2 = tf.nn.relu(z2)
        z3 = tf.matmul(a2,w3) + b3
        a3 = z3
        loss =tf.reduce_mean( (y_train - a3)**2 )
    grads = Tape.gradient(loss,[w1,b1,w2,b2,w3,b3])

    w1.assign_sub(learning_rat * grads[0])
    b1.assign_sub(learning_rat * grads[1])
    w2.assign_sub(learning_rat * grads[2])
    b2.assign_sub(learning_rat * grads[3])
    w3.assign_sub(learning_rat * grads[4])
    b3.assign_sub(learning_rat * grads[5])

    active1 = tf.reduce_mean(tf.cast(a1 > 0, tf.float32))
    active2 = tf.reduce_mean(tf.cast(a2 > 0, tf.float32))

    if epoc % 400 == 0:

        max_grad = max(
            tf.reduce_max(tf.abs(g)).numpy()
            for g in grads
            if g is not None
        )

        print(
            "epoch:", epoc,
            "loss:", loss.numpy(),
            "max_grad:", max_grad,
            "active_1:", active1,
            "active_2:", active2
        )