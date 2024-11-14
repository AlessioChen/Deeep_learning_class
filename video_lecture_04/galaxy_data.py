# from sklearn.model_selection import StratifiedShuffleSplit


# Per prima cosa dobbiamo creare una sottoclasse di `Dataset` per lavorare con
# le nostre galassie. Il modulo utils.data di pytorch mette a disposizione la
# classe base.
import torch
from torch.utils.data import Dataset, DataLoader

# Il dataset viene distribuito in formato HDF5 che è abbastanza comune per dati
# scientifici. Senza entrare nei dettagli, che potete vedere nella documentazione,
# un file HDF5 è navigabile come una sorta di file system.

class GalaxyDataset(Dataset):
    """
    see https://astronn.readthedocs.io/en/latest/galaxy10.html
    """
    def __init__(self, opts, filename, crop=True):
        import numpy as np
        import tables
        self.opts = opts
        # Tramite la libreria `pytables` possiamo aprire in lettura il file ed
        # accedere ai contenuti tramite i loro nomi. Nel caso del nostro
        # dataset, le immagini sono memorizzate nel campo `images` mentre le
        # labels sono nel campo `ans`. Ci sono altri metadati nel file che
        # ignoriamo per semplicità.
        h5 = tables.File(filename)
        X = h5.root.images[:]
        y = h5.root.ans[:]

        # Notate qui l'uso dell'indicizzazione completa [:], necessaria a forzare
        # l'effettivo caricamento in RAM dei due tensori.

        # idx = np.arange(X.shape[0])
        # np.random.shuffle(idx)
        # idx = idx[:1000]
        # np.save('X.npy', X[idx])
        # np.save('y.npy', y[idx])
        # X = np.load('X.npy')
        # y = np.load('y.npy')

        # I dati vengono normalizzati in [0,1]
        X = (X/255.).astype(np.float32)

        # Le immagini sono a colori e memorizzate in formato BHWC (channel last). Questo
        # ordine va bene con TensorFlow ma non con PyTorch, che vuole le immagini in
        # ordine BCWH (channel first). Quindi dobbiamo trasporre.
        self.X = torch.tensor(X.transpose(0,3,1,2))

        # Inoltre nelle opzioni abbiamo il flag `crop` che se vero significa
        # ritagliare un quadrato 224x224 dalle immagini originali che sono di
        # dimensione 256x256. Il crop ha due motivi:
        #
        # - L'immagine più piccola contiene comunque l'intera galassia
        #
        # - Molte architetture hanno costanti scelte sul formato 224x224 che è
        #   quello molto popolare di ImageNet, semplificando la relazione tra le
        #   costanti che usiamo adesso e quelle che trovate in molti articoli in
        #   letteratura.
        #
        # - Tra un attimo parliamo di aumentazione dove i crop vengono fatti in
        #   modo casuale.

        # Per default andiamo a prendere la parte centrale dell'immagine.
        # Viceversa imposteremo crop=False nel caso si voglia lavorare con
        # tecniche di aumentazione del dataset (che vedremo più avanti).
        if crop:
            margin = (256-224)//2
            self.X = self.X[:,:,margin:-margin,margin:-margin]

        # Infine dobbiamo costruire i vettori one-hot per i target
        self.num_classes = len(np.unique(y))

        EYE = np.eye(self.num_classes)
        y_oh = EYE[y]
        self.y = torch.tensor(y_oh)
        self.data_shape = X[0].shape

    # Per completare la classe dobbiamo scrivere altri due metodi:
    #
    # - `__len__`, che ritorna il numero di esempi (questo servirà per capire
    #   quanti minibatches ci sono in un'epoca)
    #
    # - `__get_item` che dato un indice `i` ritorna il dato i-esimo nel dataset.
    #   Trattandosi di semplice apprendimento supervisionato single-task,
    #   dobbiamo ritornare semplicemente una tupla (immagine, classe).

    def __len__(self):
        return self.X.shape[0]

    #
    def __getitem__(self, i):
        return self.X[i], self.y[i]


# Per modificare il dataloader ci basta ereditare dalla classe `GalaxyDataset`
# una nuova classe che chiamiamo `AugmentedGalaxyDataset`.

# Nel modulo `transforms` di `torchvision` sono disponibili vari algoritmi di
# trasformazione su immagini che possiamo semplicemente includere in una pipeline
# di preprocessing realizzata dalla classe `Compose`. Le trasformazioni vengono
# applicate in sequenza. A fini dimostrativi creiamo nel costruttore l'oggetto
# `augmentation_pipeline` che mette in cascata rotazioni, crop e riflessioni casuali. Nel caso
# delle galassie, qualsiasi rotazione è accettabile quindi in
# `opts.rotation_degrees` possiamo usare tutti gli angoli tra 0° e 180°. Più in
# generale, le trasformazioni hanno iperparametri che dovrebbero essere
# ottimizzati con un validation set se la background knowledge non è sufficiente.

# Infine sovraccarichiamo la `__get_item__` per invocare la pipeline di
# augmentation.

class AugmentedGalaxyDataset(GalaxyDataset):

    def __init__(self, opts, filename):
        from torchvision.transforms import v2
        super().__init__(opts, filename, crop=False)
        self.augmentation_pipeline = v2.Compose([
            v2.RandomRotation(degrees=self.opts.rotation_degrees, expand=False),
            v2.RandomCrop(size=224),
            v2.RandomHorizontalFlip(p=self.opts.horizontal_flip_probability),
            v2.RandomVerticalFlip(p=self.opts.vertical_flip_probability),
        ])

    def __getitem__(self, i):
        return self.augmentation_pipeline(self.X[i]), self.y[i]

#
# Adesso passiamo a costruire i dataloaders. Questo è molto semplice: basta
# creare un oggetto di classe `DataLoader` a partire da un oggetto di classe
# `Dataset`. In questo esempio creiamo due datasets uno di train e uno di test.
# In generale, volendo aggiustare gli iperparametri, andranno creati tre
# datasets: train, validation e test. L'argomento `lengths` consente di definire
# le proporzioni degli insiemi. Lo split non viene stratificato (ovvero non è
# garantito che le proporzioni delle diverse classi sia conservata in ciascun
# split). Questo non è problematico per datasets abbastanza grandi.
#
#
# Usiamo i parametri di default che sono sufficienti nel nostro
# caso. La classe `DataLoader` mette a disposizione vari altri strumenti, il più
# importante dei quali è forse l'argomento `collate_fn` che è una callback usata
# per assemblare i minibatches (ad esempio potrebbe essere usata per forzare una
# buona stratificazione tra le classi nel caso di datasets molto sbilanciati).


# Per dimostrare quello che accade, usiamo il dataloader appena creato per
# ottenere dei piccoli minibatches di dimensione 8 da usare a fini di
# visualizzazione.

class MakeDataLoaders():
    def __init__(self, opts, data):
        generator = torch.Generator().manual_seed(opts.seed)
        train, test = torch.utils.data.random_split(data, lengths=[1-opts.test_size, opts.test_size], generator=generator)
        self.train_dataloader = DataLoader(train, batch_size=opts.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test, batch_size=opts.batch_size, shuffle=True)


if __name__ == "__main__":
    from types import SimpleNamespace
    from ipdb import launch_ipdb_on_exception
    from helpers import plot
    opts = SimpleNamespace()
    opts.seed = 1234
    opts.test_size = 0.2
    opts.batch_size = 8
    opts.rotation_degrees = (0,180)
    with launch_ipdb_on_exception():
        data = GalaxyDataset(opts, 'Galaxy10_DECals.h5')
        decals = MakeDataLoaders(opts, data)
        train = decals.train_dataloader
        done = 0
        for X,y in train:
            print(X.shape)
            plot(
                [[xx for xx in X[:opts.batch_size//2]],[xx for xx in X[opts.batch_size//2:]]],
                [[yy for yy in y[:opts.batch_size//2]],[yy for yy in y[opts.batch_size//2:]]]
            )
            done += 1
            if done > 2:
                break

