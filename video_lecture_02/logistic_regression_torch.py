import argparse
from types import SimpleNamespace
import random
import yaml
from ipdb import launch_ipdb_on_exception
import numpy as np
import torch

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
                 device
                 ):
        torch.random.manual_seed(opts.seed)
        self.d=d
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # NOTE: REVIEW: in pytorch we may need to specify a device - a better pattern in a future video
        self.w = torch.randn(self.d, 1).to(device)
        self.b = torch.randn(1, device=device)
        self.w.requires_grad=True
        self.b.requires_grad=True
        self.optimizer = torch.optim.SGD(
            params=[self.w, self.b],
            lr=lr,
            momentum=momentum,
            nesterov=nesterov)
        # NOTE: REVIEW: Create a criterion object
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def _compute_logits(self,X):
        f = torch.matmul(X, self.w) + self.b
        return f

    def _step(self,X,y):
        f = self._compute_logits(X)
        loss = self.criterion(f, y, reduction='mean')
        # NOTE: REVIEW: This will be explained in class next week!
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()  # NOTE: Very important!!

        prediction = torch.round(torch.nn.Sigmoid()(f))
        correct = torch.eq(prediction, y).float()
        accuracy = torch.mean(correct)
        return loss,accuracy

    def fit(self, X_train, y_train, X_test, y_test):
        n,d = X_train.shape
        assert d==self.d
        random.seed(1234)
        for epoch in range(self.num_epochs):
            idx=list(range(n))
            random.shuffle(idx)
            losses,accs=[],[]
            num_batches = n//self.batch_size
            for b in range(num_batches):
                mb_idx=np.array(idx[b*self.batch_size:(b+1)*self.batch_size])
                X_mb=X_train[mb_idx]
                y_mb=y_train[mb_idx]
                loss,acc = self._step(X_mb,y_mb)
                # NOTE: data must be brough back to main RAM
                losses.append(loss.detach().numpy())
                accs.append(acc.detach().numpy())
            print(f"Epoch: {epoch+1:4d}", end=" ")
            print(f"Loss: {np.array(losses).mean():.5f}", end=" ")
            print(f"Acc: {100*np.array(accs).mean():.2f}%")

    def predict(self,X):
        n,d=X.shape
        assert d==self.d
        pred=self._compute_logits(X)
        return pred


def main(opts):
    X_train, y_train, X_test, y_test = get_adult(opts.datafile)
    y_train = np.matrix(y_train).T
    y_test = np.matrix(y_test).T
    X_train = torch.tensor(X_train).to(opts.device)
    y_train = torch.tensor(y_train).to(opts.device)
    X_test = torch.tensor(X_test).to(opts.device)
    y_test = torch.tensor(y_test).to(opts.device)

    clf = LogisticRegression(
        X_train.shape[1],
        lr=opts.lr,
        momentum=opts.momentum,
        nesterov=opts.nesterov,
        batch_size=opts.batch_size,
        num_epochs=opts.num_epochs,
        seed=opts.seed,
        device=opts.device)

    clf.fit(X_train,y_train,X_test,y_test)
    pred=clf.predict(X_test)
    acc = ((pred>0)==y_test).cpu().numpy().astype(np.float32).mean()
    print(f'Test set accuracy: {acc*100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    # NOTE: REVIEW: this allows the GPU to be used if available
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with launch_ipdb_on_exception():
        main(opts)
