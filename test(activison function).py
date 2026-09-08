import tensorflow as tf


# Model değişkenleri
w1 = tf.Variable(tf.zeros((1, 32)))
b1 = tf.Variable(tf.zeros((32,)))

w2 = tf.Variable(tf.zeros((32, 32)))
b2 = tf.Variable(tf.zeros((32,)))

w3 = tf.Variable(tf.zeros((32, 1)))
b3 = tf.Variable(tf.zeros((1,)))


# Checkpoint
checkpoint = tf.train.Checkpoint(
    w1=w1,
    b1=b1,
    w2=w2,
    b2=b2,
    w3=w3,
    b3=b3
)


# Checkpoint yöneticisi
manager = tf.train.CheckpointManager(
    checkpoint,
    "./checkpointer",
    max_to_keep=5
)


# En son checkpoint'i yükle
checkpoint.restore(manager.latest_checkpoint)


print("Checkpoint:", manager.latest_checkpoint)
print("w1:", w1)
print("w3:", w3)





#test tahmin aşaması
def predict(n):
    x_test = tf.constant([[n]],dtype=tf.float32) / 16.

    z1 = tf.matmul(x_test, w1) + b1
    a1 = tf.nn.relu(z1)

    z2 = tf.matmul(a1, w2) + b2
    a2 = tf.nn.relu(z2)


    z3 = tf.matmul(a2, w3) + b3
    a3 = z3

    prediction = a3 * 256.0

    print("Tahmin:", prediction.numpy())



predict(1)
predict(5)
predict(10)
predict(16)
predict(17)
predict(20)
predict(30)

