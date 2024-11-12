import torch
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

class ClassificationExperiment(object):

    def __init__(self, model, writer: SummaryWriter, log_interval:int = 250, lr: float = .001, device = 'cuda'):
        self.model = model
        self.writer = writer
        self.log_interval = log_interval
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    def train(self, train_loader, epoch):
        self.model.train()
        losses = []

        for step, (data, target) in enumerate(tqdm(train_loader, leave=False, desc="Running training phase")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            global_step = (len(train_loader) * train_loader.batch_size) * epoch + train_loader.batch_size * step
            
            if step % self.log_interval == 0 and step > 0:
                self.writer.add_scalar('avg_loss/train', np.mean(losses), global_step)
                 
        mean_loss = np.mean(losses)
        return losses, mean_loss

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for (data, target) in tqdm(test_loader, leave=False, desc="Running testing phase"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += torch.nn.functional.nll_loss(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        return test_loss, accuracy
    

    def fit(self, train_loader, test_loader, epochs):
        training_losses = []
        mean_losses = []
        test_losses = []
        accuracies = []

        for epoch in trange(epochs, desc="Completed epochs"):
            training_loss, mean_loss = self.train(train_loader, epoch)
            test_loss, accuracy = self.test(test_loader)
            training_losses += training_loss
            mean_losses.append(mean_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

        print("\n\033[32mTraining Completed.\033[0m")
        print(f"\033[32mFinal test accuracy: {accuracies[-1]}\033[0m\n")
            