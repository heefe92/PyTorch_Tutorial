import numpy as np
import collections
import model


from trainer import Trainer
import torch as t
import torch.optim as optim
from dataset import Dataset
from torch.utils import data as data_
import cv2,time
from config import opt

def run_train(train_verbose=False):
    pass
    dataset = Dataset()
    dataloader = data_.DataLoader(dataset, \
                                      batch_size=opt.batch_size, \
                                      shuffle=True, \
                                      # pin_memory=True,
                                      num_workers=opt.num_workers)

    testset = Dataset()
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=opt.batch_size,
                                       num_workers=opt.num_workers,
                                       shuffle=False#, \
                                       #pin_memory=True
                                       )

    my_model = model.MyModel()
    my_model = my_model.cuda()

    optimizer = optim.Adam(my_model.parameters(), lr=opt.lr)

    loss_hist = collections.deque(maxlen=500)
    epoch_loss_hist = []
    my_trainer = Trainer(my_model,optimizer,model_name='MyModel')

    best_loss = 10
    best_loss_epoch_num = 0
    num_bad_epochs = 0
    max_bad_epochs = 5

    for epoch_num in range(opt.epoch):
        my_trainer.train_mode()
        train_start_time = time.time()
        train_epoch_loss = []
        start = time.time()
        for iter_num, data in enumerate(dataloader):
            curr_loss = my_trainer.train_step(data)
            loss_hist.append(float(curr_loss))
            train_epoch_loss.append(float(curr_loss))

            if (train_verbose):
                print('Epoch: {} | Iteration: {} | loss: {:1.5f} | Running loss: {:1.5f} | Iter time: {:1.5f} | Train'
                      ' time: {:1.5f}'.format(epoch_num, iter_num, float(curr_loss), np.mean(loss_hist),
                       time.time()-start, time.time()-train_start_time))
                start = time.time()

            del curr_loss
        print('train epoch time :', time.time() - train_start_time)
        print('Epoch: {} | epoch train loss: {:1.5f}'.format(
            epoch_num, np.mean(train_epoch_loss)))

        vali_start_time = time.time()
        vali_epoch_loss = []
        my_trainer.eval_mode()

        for iter_num, data in enumerate(test_dataloader):
            curr_loss = my_trainer.get_loss(data)
            vali_epoch_loss.append(float(curr_loss))
        
            del curr_loss
        
        print('vali epoch time :', time.time() - vali_start_time)
        print('Epoch: {} | epoch vali loss: {:1.5f}'.format(
            epoch_num, np.mean(vali_epoch_loss)))
        
        if (best_loss < np.mean(vali_epoch_loss)):
            num_bad_epochs += 1
        else:
            best_loss = np.mean(vali_epoch_loss)
            best_map_epoch_num = epoch_num
            num_bad_epochs = 0
            my_trainer.model_save(epoch_num)
        if (num_bad_epochs > max_bad_epochs):
            num_bad_epochs = 0
            my_trainer.model_load(best_loss_epoch_num)
            my_trainer.reduce_lr(factor=0.1, verbose=True)

        print('best epoch num', best_loss_epoch_num)
        print('----------------------------------------')

    print(epoch_loss_hist)


if __name__ == "__main__":
    run_train(train_verbose = True)