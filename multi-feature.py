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
    xs[:, 0:1],
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


w = tf.Variable([[0.5],[3.0],[1.0],[2.3]])
b = tf.Variable(1.0)

learning_rate = 0.001


def leakly_relu(output):
    if output < 0 :
        return output * 0.001
    else:
        return output


def türev(w,b):
    predict = tf.matmul(x,w)+b
    w_grad = tf.reduce_mean( 2*(predict - y_true)*x , axis=0)
    w_grad = tf.reshape(w_grad , (4,1))
    b_grad =tf.reduce_mean( 2 * (predict - y_true)  )
    loss =tf.reduce_mean((predict - y_true)**2)
    return w_grad ,b_grad,loss ,predict






for epoc in range(100000):
    w_grad , b_grad,loss,predict = türev(w,b)

    w.assign_sub( w_grad * learning_rate)
    b.assign_sub( b_grad * learning_rate)



    if epoc % 200 == 0:
        print(f"weight {w}  bias {b}  loss {loss}  predict {predict}")

