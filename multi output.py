import tensorflow as tf
import time



X =tf.constant( [
    [2, 7, 60],
    [4, 8, 75],
    [1, 5, 40],
    [6, 8, 90],
    [3, 6, 65],
    [5, 7, 85]
], dtype=tf.float32)

X = X / tf.constant([6. , 8. , 100.])

Y =tf.constant( [
    [55, 50, 60, 58],
    [75, 70, 78, 76],
    [40, 35, 45, 42],
    [90, 88, 92, 91],
    [65, 60, 68, 64],
    [85, 82, 88, 86]
],dtype=tf.float32)

Y = Y / 100


w1 = tf.Variable(tf.random.normal((3,2)),dtype=tf.float32)
b1 = tf.Variable(tf.zeros((2,)), dtype=tf.float32)

w2 = tf.Variable(tf.random.normal((2,4)), dtype=tf.float32)
b2 = tf.Variable(tf.zeros((4,)), dtype=tf.float32)

learning_rat = 0.01
for epoc in range(2000):
    with tf.GradientTape() as Tape:
        z1 = tf.matmul(X,w1) + b1

        z2 = tf.matmul(z1,w2) + b2

        loss = tf.reduce_mean((z2 - Y)**2)
    grads = Tape.gradient(loss , [w1,b1,w2,b2])

    w1.assign_sub(learning_rat * grads[0])
    b1.assign_sub(learning_rat * grads[1])
    w2.assign_sub(learning_rat * grads[2])
    b2.assign_sub(learning_rat * grads[3])

    if epoc % 200 == 0:
        print(f"loss: {loss}   weights: {[w1,b1,w2,b2]}" )








# test bölgesi
x_train = tf.constant([[4.,7.,90.]])
x_train = x_train /  tf.constant([[6. , 8. , 100.]])

z1 = tf.matmul(x_train,w1) +b1
z2 = tf.matmul(z1,w2) +b2
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
time.sleep(2)
print("test başlatılıyor ...")
time.sleep(1)
print("\n\n\n ortam hazırlanıyor...")
time.sleep(2)
print(f"giriş değerleri: \nçalişma: 4\nuyku: 7\nnet: 90 ")
time.sleep(3)
print(f"sonucunuz {z2 * 100} \n\n\n\n\n\n")
sonuc = z2 * 100

print(f"""
╔══════════════════════════╗
║       TEST SONUCU        ║
╠══════════════════════════╣
║ Matematik   : {sonuc[0][0]:.2f}
║ Fizik       : {sonuc[0][1]:.2f}
║ İngilizce   : {sonuc[0][2]:.2f}
║ Genel       : {sonuc[0][3]:.2f}
╚══════════════════════════╝
""")
print("\n\n\n\n\n\n\n")
