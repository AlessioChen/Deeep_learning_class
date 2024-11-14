import os
from types import SimpleNamespace
import argparse
import logging

from ipdb import launch_ipdb_on_exception
import yaml
from rich.logging import RichHandler
import numpy as np
import torch

from simple_cnn import SimpleCNN
from densenet import DenseNet

def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()

def N(x):
    return x.detach().cpu().numpy()

def save_checkpoint(model, optimizer, epoch, loss, opts):
    fname = os.path.join(opts.checkpoint_dir,f'e_{epoch:05d}.chp')
    info = dict(model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch, loss=loss)
    torch.save(info, fname)
    LOG.info(f'Saved checkpoint {fname}')

def test_metrics(model, test):
    correct = []
    for Xcpu, Y in test:
        X = Xcpu.to(opts.device)
        Y = Y.to(opts.device)
        out = model(X)
        c = list(np.argmax(N(out),axis=1) == np.argmax(N(Y),axis=1))
        correct.extend(c)
    return np.mean(correct)

def train_loop(model, train, test, opts):
    import tensorflow as tf
    train_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/train')
    test_writer = tf.summary.create_file_writer(f'tensorboard/{opts.model}/test')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opts.learning_rate,
                                  weight_decay=opts.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()  # This class expects logits
    step = 0
    for epoch in range(1,opts.num_epochs+1):
        model.train()  # Sets the model in training mode
        mses = []
        correct = []
        batch_i = 0
        for Xcpu, Y in train:
            batch_i += 1
            X = Xcpu.to(opts.device)
            Y = Y.to(opts.device)
            optimizer.zero_grad()
            out = model(X)
            loss = loss_function(out, Y)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
            mses.append(N(loss))
            c = np.mean(np.argmax(N(out),axis=1) == np.argmax(N(Y),axis=1))
            correct.append(c)
            if batch_i % opts.log_every == 0:
                train_l = np.mean(mses[-opts.batch_window:])
                train_a = np.mean(correct[-opts.batch_window:])
                test_a = test_metrics(model, test)
                msg = f'{epoch:03d}.{batch_i:03d}: '
                msg += f'train: loss={train_l:1.6f}, acc={train_a:1.3f} '
                msg += f'test: acc={test_a:1.3f}'
                LOG.info(msg)
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_l, step=step)
                    tf.summary.scalar('accuracy',train_a, step=step)
                with test_writer.as_default():
                    tf.summary.scalar('accuracy',test_a, step=step)
                step += 1
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss, opts)
        scheduler.step()
        print(scheduler.get_last_lr())

def main(opts):
    from visualize_network import visualize
    from galaxy_data import GalaxyDataset, AugmentedGalaxyDataset, MakeDataLoaders
    input_size = 224
    input_data = torch.randn(opts.batch_size, 3, input_size, input_size)
    num_classes = 10
    if opts.model == 'simple':
        model = SimpleCNN(num_filters=opts.simple_num_filters,
                          mlp_size=opts.simple_mlp_size,
                          num_classes=num_classes)
    elif opts.model == 'densenet':
        model = DenseNet(input_size=input_size,
                         num_layers=opts.densenet_num_layers,
                         k=opts.densenet_k,
                         num_dense_blocks=opts.densenet_num_dense_blocks,
                         num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model type {opts.model}')
    visualize(model, 'DenseNet', input_data)
    model = model.to(opts.device)
    if opts.augmentation:
        data = AugmentedGalaxyDataset(opts, 'Galaxy10_DECals.h5')
    else:
        data = GalaxyDataset(opts, 'Galaxy10_DECals.h5')
    decals = MakeDataLoaders(opts, data)
    train = decals.train_dataloader
    test = decals.test_dataloader

    train_loop(model, train, test, opts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help='YAML Configuration file')
    opts = yaml.load(open(parser.parse_args().config), Loader=yaml.Loader)
    opts = SimpleNamespace(**opts)
    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with launch_ipdb_on_exception():
        main(opts)

