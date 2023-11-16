from module import DataLoad,train_print_progress
import torch

class On_Train:
    def __init__(self, model, optimizer, data_loader, device, model_save_dir):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.model_save_dir = model_save_dir


    def custom_callbacks(self):
        pass


    def run(self, epoch, is_eval):
        for i,(images,targets) in enumerate(self.data_loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            self.optimizer.step()

            train_print_progress(i, len(self.data_loader), loss.item())
        torch.save(self.model.state_dict(), f'{self.model_save_dir}/model_num_{epoch}.pt')
        print("model_saved")