import tensorflow as tf


# Model değişkenlerimiz
n = 64
w1 = tf.Variable(tf.zeros((1, n)))
b1 = tf.Variable(tf.zeros((n,)))

w2 = tf.Variable(tf.zeros((n, n)))
b2 = tf.Variable(tf.zeros((n,)))

w3 = tf.Variable(tf.zeros((n, 1)))
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
    x_test = tf.constant(n, dtype=tf.float32)
    x_test = tf.reshape(x_test, (-1, 1))

    z1 = tf.matmul(x_test / 30.0, w1) + b1
    a1 = tf.nn.relu(z1)

    z2 = tf.matmul(a1, w2) + b2
    a2 = tf.nn.relu(z2)

    z3 = tf.matmul(a2, w3) + b3
    prediction = z3 * 900.0

    for x_value, pred in zip(x_test.numpy().flatten(), prediction.numpy().flatten()):
        print(
            f"x={x_value:.1f} | "
            f"gerçek={x_value**2:.2f} | "
            f"tahmin={pred:.2f}"
        )


predict(1.5)
predict(5.5)
predict(10.5)
predict(16.5)
predict(17.5)
predict(20.5)
predict(25.5)
predict(30.5)


test_values = []

for i in range(1, 30):
    for j in range(1, 10):
        test_values.append(i + j / 10)

predict(test_values)

