import torch

class Trainer():
    def __init__(self, my_model, optimizer, model_name):
        self.my_model=my_model
        self.optimizer=optimizer
        self.model_name = model_name
        self.my_model.train()
        self.my_model.use_preset(isTraining=True)

    def train_mode(self,freeze_num=0):
        self.my_model.use_preset(isTraining=True)
        self.my_model.freeze_bn()
        self.model_freeze(freeze_num)

    def eval_mode(self):
        self.my_model.use_preset(isTraining=False, preset='evaluate')
        self.my_model.eval()
        for child in self.my_model.children():
            for param in child.parameters():
                param.requires_grad = False
    def train_step(self, data):
        self.optimizer.zero_grad()
        loss = self.my_model(
            [data[0].cuda(), data[1].cuda()])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.my_model.parameters(), 0.1)
        self.optimizer.step()

        return loss.item()
        
    def get_loss(self, data):
        loss = self.my_model(
            [data[0].cuda(), data[1].cuda()])
        return loss.item()

    def model_save(self,epoch_num):
        torch.save(self.my_model.state_dict(), 'Weights/'+self.model_name+'_{}.pt'.format(epoch_num))

    def model_load(self,epoch_num):
        self.my_model.load_state_dict(torch.load('Weights/'+self.model_name+'_{}.pt'.format(epoch_num)))

    def reduce_lr(self,factor=0.1,verbose=True):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * factor
            param_group['lr'] = new_lr
            if verbose:
                print('reducing learning rate'
                        ' of group {} to {:.4e}.'.format( i, new_lr))

    def model_freeze(self,freeze_num):
        child_count = 0
        for child in self.my_model.children():
            if(child_count < freeze_num):
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_count+=1
