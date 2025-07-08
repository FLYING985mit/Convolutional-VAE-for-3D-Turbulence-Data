import torch
from tqdm import tqdm
from loss import loss_function
import pdb
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='train.log',
                    filemode='w')

logging.addLevelName(25,"Mu")
def Mu(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)

logging.Logger.mu = Mu

mu=logging.getLogger(__name__)

def train(epoch, model, train_loader, kl_weight, optimizer, device, scheduler, args):
    """
    Mini-batch training.
    """

    model.train()
    train_total_loss = 0
    train_BCE_loss = 0
    train_KLD_loss = 0

    logging.info("entered batch training")
    print("train device:", device)
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):

        # move data into GPU tensors
        data = data.to(device, dtype=torch.float)

        # reset gradients
        optimizer.zero_grad()

        # call CVAE model
        # feeding 3D volume to Conv3D: https://discuss.pytorch.org/t/feeding-3d-volumes-to-conv3d/32378/6
        recon_batch, mu, logvar, _ = model(data)
        logging.info(f"Epoch: {epoch} Batch {batch_idx}: mu : {mu}\nlogvar : {logvar}")

        # compute batch losses
        total_loss, BCE_loss, KLD_loss = loss_function(recon_batch, data, mu, logvar, kl_weight)

        train_total_loss += total_loss.item()
        train_BCE_loss += BCE_loss.item()
        train_KLD_loss += KLD_loss.item()

        # compute gradients and update weights
        total_loss.backward()
        optimizer.step()

        # schedule learning rate
        scheduler.step()

    train_total_loss /= len(train_loader.dataset)
    train_BCE_loss /= len(train_loader.dataset)
    train_KLD_loss /= len(train_loader.dataset)

    return train_total_loss, train_BCE_loss, train_KLD_loss
