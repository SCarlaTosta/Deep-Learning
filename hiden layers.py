import tensorflow as tf


xs = tf.constant([
    [1.0, 2.0, 3.0],
    [2.0, 1.0, 4.0],
    [3.0, 3.0, 2.0],
    [4.0, 2.0, 5.0],
    [5.0, 4.0, 1.0],
    [6.0, 3.0, 4.0],
    [7.0, 5.0, 2.0],
    [8.0, 4.0, 3.0],
])

x = tf.concat([
    xs[:, 0:1],                                 # (8,4)
    xs[:, 1:2],
    xs[:, 1:2]**2,
    xs[:, 2:3]],axis=1

)


y_true = (
    0.5 * x[:, 0:1]
    + 2.0 * x[:, 1:2]
    - 0.3 * x[:, 2:3]
    + 1
)

w1 = tf.Variable(tf.random.normal((4, 3), dtype=tf.float32))
b1 = tf.Variable(tf.zeros((3,)))

w2 = tf.Variable(tf.random.normal((3, 1), dtype=tf.float32))
b2 = tf.Variable(tf.random.normal((1,)))



learning_rat = 0.0001
for epoc in range(100000):
    with tf.GradientTape() as tape:


        z1 = tf.matmul(x,w1) + b1     # (8,3)


        a1 = tf.nn.relu(z1)           # (8,3)




        z2 = tf.matmul(a1,w2)+b2
        loss =tf.reduce_mean( (y_true - z2)**2)#(8,1)
    grads = tape.gradient(loss,[w1,b1,w2,b2])

    w1.assign_sub(grads[0] * learning_rat)
    b1.assign_sub(grads[1]* learning_rat)
    w2.assign_sub(grads[2]* learning_rat)
    b2.assign_sub(grads[3]* learning_rat)

    if epoc % 200 == 0:
        print(f"loss: {loss} predict: {z2}  w1: {w1} b1: {b1} w2: {w2} b2: {b2}")




x_new = tf.constant([
    [9.0, 6.0, 4.0]
], dtype=tf.float32)

x_new = tf.concat([
    x_new[:, 0:1],
    x_new[:, 1:2],
    x_new[:, 1:2] ** 2,
    x_new[:, 2:3]
], axis=1)


z1_new = tf.matmul(x_new, w1) + b1
a1_new = tf.nn.relu(z1_new)

z2_new = tf.matmul(a1_new, w2) + b2

print("Tahmin:", z2_new)