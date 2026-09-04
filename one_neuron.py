import tensorflow as tf



x_s = tf.constant([[500 , 3, 4],
                 [1000 , 8 , 2],
                 [750, 5 , 5],
                 [1200 , 10 , 2],
                 [1500 , 12  , 1],
                 [300 , 3 , 12]],dtype=tf.float32)

x = x_s / tf.reduce_max(x_s,axis=0)


y_trues = tf.constant([[550000.0],
                 [1100000.0],
                 [820000.0],
                 [1320000.0],
                 [1620000.0],
                 [350000.0]], dtype=tf.float32)

y_true = y_trues / 1000000.0



w = tf.Variable([[0.5],[3.0],[1.0]])
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
    w_grad = tf.reshape(w_grad , (3,1))
    b_grad =tf.reduce_mean( 2 * (predict - y_true)  )
    loss =tf.reduce_mean((predict - y_true)**2)
    return w_grad ,b_grad,loss ,predict






for epoc in range(100000):
    w_grad , b_grad,loss,predict = türev(w,b)

    w.assign_sub( w_grad * learning_rate)
    b.assign_sub( b_grad * learning_rate)



    if epoc % 200 == 0:
        print(f"weight {w}  bias {b}  loss {loss}  predict {predict}")

