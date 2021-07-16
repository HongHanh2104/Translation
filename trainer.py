import torch 
#from torchnet import meter
import numpy as np
from tqdm import tqdm  
from datetime import datetime
import os 
import json

from loggers import TensorboardLogger
from utils.utils import idx_to_word

class Trainer():
    def __init__(self, model, 
                 device,
                 data,
                 iterator,
                 loss, 
                 metric, 
                 optimizer,
                 scheduler,
                 config):

        self.config = config
        self.device = device
        
        self.model = model
        self.train_iter, self.val_iter = iterator
        self.data = data

        self.loss = loss
        self.metric = metric
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grads = self.config['dataset']['train']['clip_grads']
        
        # Train ID 
        self.train_id = self.config['id']
        self.train_id += ('-' + datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        print(self.train_id)

        self.save_dir = os.path.join('checkpoints', self.train_id)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # Logger
        self.tsboard = TensorboardLogger(path=self.save_dir)

        # Get arguments
        self.nepochs = self.config['trainer']['nepochs']
        self.log_step = self.config['trainer']['log_step']
        self.val_step = self.config['trainer']['val_step']
        
        self.best_loss = np.inf
        self.best_metric = 0.0
        self.val_loss = list()
        

    def train_epoch(self, epoch, iterator):
        print('Training........')

        # 0: Record loss during training process
        running_loss = []
        running_tokens = []
        total_loss = []
        total_tokens = []
            
        # Switch model to training mode
        self.model.train()

        # Setup progress bar
        progress_bar = tqdm(iterator)
        for i, batch in enumerate(progress_bar):
            # 1: Load sources, inputs, and targets
            src = batch.src.to(self.device)
            trg = batch.trg.to(self.device)

            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            
            # 3: Get network outputs
            out = self.model(src, trg[:, :-1])
            trg = trg[:, 1:]
            
            #out_reshape = out.contiguous().view(-1, out.shape[-1])
            #trg = trg.contiguous().view(-1)

            # 4: Calculate the loss
            loss, ntokens = self.loss(out, trg)
            
            # 5: Calculate gradients
            loss.backward()
            
            # 6: Performing backpropagation
            if self.clip_grads:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            with torch.no_grad():
                # 7: Update loss
                running_loss.append(loss.item())
                running_tokens.append(ntokens)
                total_loss.append(loss.item())
                total_tokens.append(ntokens)
                
                # Update loss every log_step or at the end
                if i % self.log_step == 0 or (i + 1) == len(iterator):
                    #print(f'Epoch step {i}, Loss: {sum(running_loss) / sum(running_tokens)}, Step: {epoch * len(iterator) + i}')
                    self.tsboard.update_loss(
                            'train', 
                            sum(running_loss) / sum(running_tokens), 
                            epoch * len(iterator) + i)
                    running_loss.clear()
                    running_tokens.clear()

                # 8: Update metric
                #outs = outs.detach()
                #trgs = trgs.detach()
                #metric, _ = self.metric.calculate(outs, trgs)
                #running_metric.append(metric)
        
        print('++++++++++++++ Training result ++++++++++++++')
        avg_loss = sum(total_loss) / sum(total_tokens)
        print('Loss: ', avg_loss) 
        return avg_loss  
    
    @torch.no_grad()
    def val_epoch(self, epoch, iterator):
        print('Evaluating........')

        # 0: Record loss during training process
        total_loss = []
        total_tokens = []
        batch_bleu = []
        
        # Switch model to training mode
        self.model.eval()

        # Setup progress bar
        progress_bar = tqdm(iterator)
    
        for i, batch in enumerate(progress_bar):
            # 1: Load sources, inputs, and targets
            src = batch.src.to(self.device)
            trg = batch.trg.to(self.device)
                
            # 2: Get network outputs
            out = self.model(src, trg[:, :-1])
            trg = trg[:, 1:]
            
            # 3: Calculate the loss
            loss, ntokens = self.loss(out, trg)
            
            # 4: Update loss
            total_loss.append(loss.item())
            total_tokens.append(ntokens)
            
            # 5: Update metric
            out = out.detach()
            trg = trg.detach()
                
            total_bleu = []
                
            for j in range(src.size()[0]):
                trg_words = idx_to_word(x=batch.trg[j], vocab=self.data.target.vocab)
                out_words = out[j].max(dim=1)[1]
                out_words = idx_to_word(x=out_words, vocab=self.data.target.vocab)
                bleu = self.metric.get_bleu(hypothesis=out_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)
            
            score_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(score_bleu)

        print("++++++++++++++ Evaluation result ++++++++++++++")
        loss = sum(total_loss) / sum(total_tokens)
        print('Loss: ', loss)
        accuracy = sum(batch_bleu) / len(batch_bleu)
        print('Accuracy: ', accuracy)
        
        # Upload tensorboard
        self.tsboard.update_loss('val', loss, epoch)
        self.tsboard.update_metric('val', accuracy, epoch)
        return loss, accuracy

    def train(self):
        for epoch in range(self.nepochs):
            print('\nEpoch {:>3d}'.format(epoch))
            print('-----------------------------------')

            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group['lr'], epoch)

            # Train phase
            train_loss = self.train_epoch(epoch, self.train_iter)
            
            # Eval phase
            val_loss, bleu = self.val_epoch(epoch, self.val_iter)
            
            if epoch > self.config['optimizer']['warmup']:
                self.scheduler.step(val_loss)
                        
            if (epoch + 1) % self.val_step == 0:
                # Save weights
                self._save_model(epoch, train_loss, val_loss, bleu)


    def _save_model(self, epoch, train_loss, val_loss, bleu):
        
        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }

        loss_save_format = 'val_loss={val_loss:<.3}.pth'
        loss_save_format_filename = loss_save_format.format(
            val_loss=val_loss
        )
        
        metric_save_format = 'bleu={bleu:<.3}.pth'
        metric_save_format_filename = metric_save_format.format(
            bleu=bleu
        )

        if val_loss < self.best_loss:
            print(
                f'Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights ...'
            )
            torch.save(data, os.path.join(self.save_dir, 'best_loss-' + loss_save_format_filename))
            self.best_loss = val_loss
        else:
            print(f'Loss is not improved from {self.best_loss: .6f}.')
        
        if bleu > self.best_metric:
            print(
                f'Accuracy is improved from {self.best_metric: .6f} to {bleu: .6f}. Saving weights ...'
            )
            torch.save(data, os.path.join(self.save_dir, 'best_bleu-' + metric_save_format_filename))
            self.best_metric = bleu
        else:
            print(f'Accuracy is not improved from {self.best_metric: .6f}.')

        

