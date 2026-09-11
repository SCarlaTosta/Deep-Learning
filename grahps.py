import tensorflow as tf
import matplotlib.pyplot as plt



n = 64

w1 = tf.Variable(tf.zeros((1, n)))
b1 = tf.Variable(tf.zeros((n,)))

w2 = tf.Variable(tf.zeros((n, n)))
b2 = tf.Variable(tf.zeros((n,)))

w3 = tf.Variable(tf.zeros((n, 1)))
b3 = tf.Variable(tf.zeros((1,)))


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

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    print("Model yüklendi:", manager.latest_checkpoint)
else:
    print("Checkpoint bulunamadı!")
    exit()




test_values = []

for i in range(-30, 60):
    for j in range(1, 10):
        test_values.append(i + j / 10)




x_test = tf.constant(test_values, dtype=tf.float32)


x_normalized = tf.reshape(x_test, (-1, 1)) / 30.0



z1 = tf.matmul(x_normalized, w1) + b1
a1 = tf.nn.relu(z1)


z2 = tf.matmul(a1, w2) + b2
a2 = tf.nn.relu(z2)



z3 = tf.matmul(a2, w3) + b3


predictions = (z3 * 900.0).numpy().flatten()



real_values = x_test.numpy() ** 2



plt.plot(
    x_test.numpy(),
    real_values,
    label="Gerçek x²"
)

plt.plot(
    x_test.numpy(),
    predictions,
    label="Model tahmini"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title("Model ve Gerçek x²")

plt.legend()
plt.grid()

plt.show()