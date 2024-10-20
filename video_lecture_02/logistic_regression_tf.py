import argparse
from types import SimpleNamespace

import random
import yaml

from ipdb import launch_ipdb_on_exception
import numpy as np
import tensorflow as tf

from data import get_adult

class LogisticRegression():
    def __init__(self,
                 d,           # Dimension of the input vector
                 lr,          # learning rate
                 momentum,    # momentum
                 nesterov,    # NAG (if True) or classic momentum (False)
                 batch_size,  # Batch size
                 num_epochs,  # number of cycles through the whole training set
                 seed,        # !=0 to reproduce from run to run
                 ):
        pass
        # lock the seed for reproducibility
        tf.random.set_seed(seed)
        self.d = d
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # Declare parameters as tensorflow variables
        self.w = tf.Variable(tf.random.normal(shape=[d, 1]),name='w')
        self.b = tf.Variable(tf.random.normal(shape=[1, 1]),name='b')
        self.optimizer = tf.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=nesterov)

    def _compute_logits(self, X):
        f = tf.matmul(X, self.w) + self.b
        return f

    def _step(self, X, y):
        with tf.GradientTape() as tape:
            f = self._compute_logits(X)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=f, labels=y)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, [self.w, self.b])
        self.optimizer.apply_gradients(zip(gradients, [self.w, self.b]))
        prediction = tf.round(tf.sigmoid(f))
        correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)
        return loss,accuracy

    def _metrics(self, X, y):
        f = self._compute_logits(X)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=f, labels=y)
        loss = tf.reduce_mean(loss)
        prediction = tf.round(tf.sigmoid(f))
        correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
        accuracy = tf.reduce_mean(correct)
        return loss,accuracy

    def fit(self, X_train, y_train, X_test, y_test):
        n,d = X_train.shape
        assert d==self.d
        random.seed(1234)
        train_summary_writer = tf.summary.create_file_writer('tensorboard/log_regr/train')
        test_summary_writer = tf.summary.create_file_writer('tensorboard/log_regr/test')
        for epoch in range(self.num_epochs):
            idx=list(range(n))
            random.shuffle(idx)
            losses,accs=[],[]
            num_batches = n//self.batch_size
            for b in range(num_batches):
                mb_idx=np.array(idx[b*self.batch_size:(b+1)*self.batch_size])
                X_mb=X_train[mb_idx]
                y_mb=np.matrix(y_train[mb_idx]).T
                loss,acc = self._step(X_mb,y_mb)
                losses.append(loss)
                accs.append(acc)
            print(f"Epoch: {epoch+1:4d}", end=" ")
            print(f"Loss: {np.array(losses).mean():.5f}", end=" ")
            print(f"Acc: {100*np.array(accs).mean():.2f}%")
            with train_summary_writer.as_default():
                loss,acc = self._metrics(X_train,np.matrix(y_train).T)
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy',acc, step=epoch)
            with test_summary_writer.as_default():
                loss,acc = self._metrics(X_test,np.matrix(y_test).T)
                tf.summary.scalar('loss', loss, step=epoch)
                tf.summary.scalar('accuracy',acc, step=epoch)

    def predict(self, X_np):
        n,d = X_np.shape
        assert d == self.d
        pred=self._compute_logits(X_np)
        return (pred>0).numpy().flatten()

def main(opts):
    X_train, y_train, X_test, y_test = get_adult(opts.datafile)
    clf = LogisticRegression(
        X_train.shape[1],
        lr=opts.lr,
        momentum=opts.momentum,
        nesterov=opts.nesterov,
        batch_size=opts.batch_size,
        num_epochs=opts.num_epochs,
        seed=opts.seed)
    clf.fit(X_train,y_train,X_test,y_test)
    pred=clf.predict(X_test)
    print(f'Test set accuracy: {np.mean(pred==y_test)*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    with launch_ipdb_on_exception():
        main(opts)
