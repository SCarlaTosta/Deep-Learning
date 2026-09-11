from platform import win32_ver

import tensorflow as tf



x = []

for i in range(1, 31):
    x.append([i])

x = tf.constant(x,dtype=tf.float32)

y = x ** 2

x_train = x / 30.0
y_train = y / 900.0

tf.random.set_seed(12)
n = 64
w1 = tf.Variable(tf.random.normal((1,n),stddev=0.1))
b1 = tf.Variable(tf.zeros((n,)))

w2 = tf.Variable(tf.random.normal((n,n),stddev=0.1))
b2 = tf.Variable(tf.zeros((n,)))

w3 = tf.Variable(tf.random.normal((n,1),stddev=0.1))
b3 = tf.Variable(tf.zeros((1,)))




optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

#optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9) #



checkpoint = tf.train.Checkpoint(
    w1=w1,
    b1=b1,
    w2=w2,
    b2=b2,
    w3=w3,
    b3=b3
)

manager = tf.train.CheckpointManager(
    checkpoint,
    "./checkpointer",
    max_to_keep=5
)




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

### buraya optimizer çağrılcak
    optimizer.apply_gradients(zip( grads,[w1,b1,w2,b2,w3,b3]))


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
        manager.save()