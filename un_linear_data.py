import tensorflow as tf



x = tf.constant([[
    -10., -9.8, -9.6, -9.4, -9.2, -9., -8.8, -8.6, -8.4, -8.2,
    -8., -7.8, -7.6, -7.4, -7.2, -7., -6.8, -6.6, -6.4, -6.2,
    -6., -5.8, -5.6, -5.4, -5.2, -5., -4.8, -4.6, -4.4, -4.2,
    -4., -3.8, -3.6, -3.4, -3.2, -3., -2.8, -2.6, -2.4, -2.2,
    -2., -1.8, -1.6, -1.4, -1.2, -1., -0.8, -0.6, -0.4, -0.2,
     0.,  0.2,  0.4,  0.6,  0.8,  1.,  1.2,  1.4,  1.6,  1.8,
     2.,  2.2,  2.4,  2.6,  2.8,  3.,  3.2,  3.4,  3.6,  3.8,
     4.,  4.2,  4.4,  4.6,  4.8,  5.,  5.2,  5.4,  5.6,  5.8,
     6.,  6.2,  6.4,  6.6,  6.8,  7.,  7.2,  7.4,  7.6,  7.8,
     8.,  8.2,  8.4,  8.6,  8.8,  9.,  9.2,  9.4,  9.6,  9.8,
    10.
]], dtype=tf.float32)

y = tf.constant([[
    25.0, 23.1, 21.4, 19.6, 18.0, 16.5, 15.1, 13.8, 12.6, 11.5,
    10.4,  9.4,  8.5,  7.6,  6.8,  6.0,  5.3,  4.7,  4.1,  3.6,
     3.1,  2.7,  2.3,  2.0,  1.7,  1.5,  1.3,  1.1,  1.0,  0.9,
     0.8,  0.8,  0.9,  1.0,  1.2,  1.4,  1.7,  2.0,  2.4,  2.9,
     3.4,  4.0,  4.7,  5.4,  6.2,  7.0,  7.9,  8.8,  9.8, 10.8,
    11.8, 12.9, 14.0, 15.1, 16.3, 17.5, 18.7, 19.9, 21.1, 22.3,
    23.5, 24.7, 25.9, 27.1, 28.3, 29.5, 30.7, 31.9, 33.1, 34.3,
    35.5, 36.7, 37.9, 39.1, 40.3, 41.5, 42.7, 43.9, 45.1, 46.3,
    47.5, 48.7, 49.9, 51.1, 52.3, 53.5, 54.7, 55.9, 57.1, 58.3,
    59.5, 60.7, 61.9, 63.1, 64.3, 65.5, 66.7, 67.9, 69.1, 70.3,
    71.5
]], dtype=tf.float32)

x = tf.transpose(x)
y = tf.transpose(y)

print(x.shape)
print(y.shape)

w1 = tf.Variable(tf.random.normal((1,10)))
b1 = tf.Variable(tf.ones((10,)))

w2 = tf.Variable(tf.random.normal((10,1)))
b2 = tf.Variable(tf.ones((1,)))



checkpoint = tf.train.Checkpoint(
    w1=w1,
    b1=b1,
    w2=w2,
    b2=b2
)



manager = tf.train.CheckpointManager(
    checkpoint,
    "./checkpoints",
    max_to_keep=2
)

learning_rat = 0.0001


print("x:", x.shape)
print("y:", y.shape)
for epoc in range(200000):
    with tf.GradientTape() as Tape:
        z1 = tf.matmul(x,w1) + b1

        a1 = tf.nn.relu(z1)
        z2 = tf.matmul(a1,w2) + b2

        loss =tf.reduce_mean( (z2 - y)**2 )
    grads = Tape.gradient(loss,[w1,b1,w2,b2])

    w1.assign_sub(grads[0] * learning_rat)
    b1.assign_sub(grads[1] * learning_rat)
    w2.assign_sub(grads[2] * learning_rat)
    b2.assign_sub(grads[3] * learning_rat)


    if epoc %200 == 0:
        print(f" w1:   {w1}   b1:    {b1}      w2:       {w2}         b2:       {b2}       \n loss : {loss}     ")
        manager.save()




