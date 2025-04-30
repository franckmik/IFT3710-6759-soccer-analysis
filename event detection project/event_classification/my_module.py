import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset,DataLoader, random_split,Subset
import torch.optim as optim
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import time
from sklearn.metrics import accuracy_score
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torchvision.models import efficientnet_b0





class StratifiedSplit:
  def __init__(self, test_size=.1, random_state=42):
    self.splitter=StratifiedShuffleSplit(
      n_splits=1,
      test_size=test_size,
      random_state=random_state
    )
  def __call__(self,dataset):
    #extraction des labels
    if isinstance(dataset,torch.utils.data.Subset):
      labels=[dataset.dataset.targets[i] for i in dataset.indices]
    elif hasattr(dataset,'targets'):
      labels= dataset.targets
    else:
      raise ValueError('Impossible de trouver les labels du dataset')
    train_idx, val_idx= next(
      self.splitter.split(
        X=range(len(dataset)),
        y=labels
      )
    )
    return Subset(dataset,train_idx),Subset(dataset,val_idx)
    


class SoccerEventDataset(Dataset):
  """
    - Custom Dataset class for initiating dataset classes
  """
  def __init__(self,root_dir,transform=None):
    """
    Args:
      - root_dir(string)= images folder
      -transform(callable)= transformation to apply on images
    """
    self.root_dir=root_dir
    self.transform=transform

   # sorted list for classes
    self.classes=sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
    self.class_to_idx={cls_name:i for i ,cls_name in enumerate(self.classes)}

  #images paths and labels
    self.image_paths=[]
    self.labels=[]

    for class_name in self.classes:
      class_idx= self.class_to_idx[class_name]
      class_path= os.path.join(root_dir,class_name)
      try:
        for file in os.scandir(class_path):
          if self._is_valid_image(file.path):
            self.image_paths.append(file.path)
            self.labels.append(class_idx)
      except FileNotFoundError:
        print(f"Warning: Missing class directory {class_path}")
        continue

  def _is_valid_image(self,path):
    """
      - Image validation using file headers
    """
    valid_extensions=('.png','.jpg','.jpeg','.bmp','.gif')
    return(
        os.path.isfile(path) and
        path.lower().endswith(valid_extensions) and
        self._is_actual_image(path)
    )
  def _is_actual_image(self,path):
    """
     - helper function for image validation
    """
    try:
      Image.open(path).verify()
      return True
    except (IOError,SyntaxError):
      return False

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self,idx):
    try:
      image= Image.open(self.image_paths[idx]).convert('RGB')
      label=self.labels[idx]
      if self.transform:
        image=self.transform(image)
      return image, label
    except Exception as e:
      print(f"Error loading {self.image_paths[idx]} :{str(e)}")
      return self._get_fallback_item()
  def _get_fallback_item(self):
    """
     return a modify vector as a placeholder
    """
    return torch.zeros(3,224,224)
    
##############################################################################
##############################################################################
##############################################################################
class SoccerEventClassifier(nn.Module):
  """
  - classification model as a class 
  """
  def __init__(self):
    super().__init__()
    #loading pretrained model
    #self.backbone= efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    self.backbone = efficientnet_b0(weights=weights)

    #adapt the classifier for 9 outputs
    num_features= self.backbone.classifier[1].in_features

    #classes in sorted order
    self.class_names=[
        'Cards',
        'Center',
        'Corner',
        'Free-Kick',
        'Left',
        'Penalty',
        'Right',
        'Tackle',
        'To-Subtitue']
    self.backbone.classifier= nn.Sequential(
        # first dense Layer
        nn.Linear(num_features,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(.3),
        

        #second dense Layer
        nn.Linear(512,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(.3),

        #final classification layer
        nn.Linear(512,len(self.class_names))
    )
  def forward(self,x):
    return self.backbone(x)
  
  ##############################################################################
##############################################################################
##############################################################################



def train_model(
    model,
    batch_size,
    num_epochs,
    learning_rate,
    device,
    train_loader,
    val_loader,
    loss_type='cross_entropy',
    class_weights=torch.tensor([1.0, 1.0, 1.0, 2.0, 1.5, 1.0, 1.5, 1.0, 1.0]),
    use_scheduler=True,
    clip_grad=1.0,
    early_stop_patience=5,
    use_mixup=True,
    mixup_alpha=.2
):
  """
  fonction d'entrainement qui prends en arguments:
  model: model a entrainer,
  batch_size(int): taille des batches,
  num_epochs(int): nombre d'iterations
  learning_rate(float),
  device(string): cuda or cpu,
  train_loader(DataLoader),
  val_loader(DataLoader)
  """
  #configuration de la perte
  if loss_type=='focal':
    criterion=FocalLoss(alpha=class_weights,gamma=2)
  else:
    criterion= nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
  
  #optimizer
  optimizer= optim.AdamW(model.parameters(),lr=learning_rate, weight_decay=.05)

  #learning rate scheduler
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',
                                                   patience=2,factor=.5) if use_scheduler else None
  #metrics
  metrics={
    'train_loss':[],
    'val_loss':[],
    'train_acc':[],
    'val_acc':[],
    'lrs':[]
  }

  best_acc=.0
  no_improve_epochs=0

  #Training loop
  for epoch in range(num_epochs):
    model.train()
    running_loss=.0
    correct_train=0
    total_train=0
    start_time= time.time()
    for images, labels in train_loader:
      images, labels= images.to(device), labels.to(device)

      #mixup augmentation
      apply_mixup=use_mixup and np.random.rand() <.5
      if apply_mixup:
        images, labels_a,labels_b,lam= mixup_data(images,labels,mixup_alpha)
        images= images.to(device)
        labels_a= labels_a.to(device)
        labels_b= labels_b.to(device)

      optimizer.zero_grad()
      outputs=model(images)

      #mixup loss calculation
      if apply_mixup:
        loss= mixup_criterion(criterion,outputs,labels_a,labels_b,lam)
      else:
        loss=criterion(outputs,labels)

      #gradient handling
      loss.backward()
      if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)
      optimizer.step()

      #metrics
      _,pred= torch.max(outputs.data,1)
      total_train+=labels.size(0)
      correct_train+=(pred==labels).sum().item()
      running_loss+=loss.item()*images.size(0)

    #validation
    val_loss, val_acc= validate_model(model,criterion,val_loader,device)

      #update metrics
    epoch_train_loss= running_loss/total_train
    epoch_train_accuracy= correct_train/total_train
    metrics['train_loss'].append(epoch_train_loss)
    metrics['train_acc'].append(epoch_train_accuracy)
    metrics['val_loss'].append(val_loss)
    metrics['val_acc'].append(val_acc)
    metrics['lrs'].append(optimizer.param_groups[0]['lr'])

      #schedulerstep
    if use_scheduler and scheduler:
      scheduler.step(val_acc)
      
      #logging
    epoch_time= time.time()-start_time
    print(f'{epoch+1}/{num_epochs} |' 
    f'Train Loss:{epoch_train_loss:.4f} | Valid Loss {val_loss:.4f} |'
    f'Train acc:{epoch_train_accuracy: .4f} | Valid Acc {val_acc:.4f} | '
    f'LR:{optimizer.param_groups[0]["lr"]:.2e} |'
    f'Time:{epoch_time//60:.0f}m {epoch_time%60:.0f}s'
    )
      #save the best model
    if val_acc >best_acc:
      best_acc=val_acc
      torch.save(model.state_dict(),'best_model_v4.pth')
      no_improve_epochs=0
    else:
      no_improve_epochs+=1
      
      #Early stopping
    if no_improve_epochs>= early_stop_patience:
      print(f'Early stopping after {epoch+1} epochs!')
      break
  print(f'Training completed: best validation accuracy{best_acc:.4f}')
  return metrics

##########################################################################
##########################################################################
#########################################################################
def validate_model(model,criterion, val_loader,device):
  model.eval()
  total_loss=.0
  correct=0
  total=0
  with torch.no_grad():
    for images, labels in val_loader:
      images,labels= images.to(device),labels.to(device)
      outputs=model(images)
      loss=criterion(outputs,labels)

      total_loss+=loss.item()*images.size(0)
      _,pred= torch.max(outputs.data,1)
      total+=labels.size(0)
      correct+=(pred==labels).sum().item()
  return total_loss/total, correct/total
##########################################################################
##########################################################################
#########################################################################

class_weights = torch.tensor([1.0, 1.0, 1.0, 2.5, 1.5, 1.0, 1.5, 1.0, 1.0])

class FocalLoss(nn.Module):
  def __init__(self,alpha=class_weights,gamma=2, reduction='mean'):
    super().__init__()
    self.alpha=alpha
    self.gamma= gamma
    self.reduction=reduction
  def forward(self,inputs,targets):
    self.alpha= self.alpha.to(inputs.device)
    ce_loss=nn.functional.cross_entropy(inputs,targets,reduction='none',weight=self.alpha)
    pt=torch.exp(-ce_loss)
    loss= (1-pt)**self.gamma*ce_loss
    if self.reduction=='mean':
      return loss.mean()
    elif self.reduction=='sum':
      return loss.sum()
    return loss
  ##########################################################################
##########################################################################
#########################################################################

def mixup_data(x,y,alpha=.2):
  if alpha>0:
    lam= float(np.random.beta(alpha,alpha))
  else:
    lam=1
  batch_size= x.size(0)
  index= torch.randperm(batch_size).to(x.device)
  mixed_x= lam*x+(1-lam)*x[index,:]
  y_a,y_b=y,y[index]
  return mixed_x,y_a,y_b,lam

##########################################################################
##########################################################################
#########################################################################

def mixup_criterion(criterion,pred,y_a,y_b,lam):
  return lam*criterion(pred,y_a)+(1-lam)*criterion(pred,y_b)




##########################################################################
##########################################################################
#########################################################################
def plot_model(history):
  train_loss=history['train_loss']
  train_acc= history['train_acc']
  val_loss= history['val_loss']
  val_acc=history['val_acc']
  lrs= history['lrs']
  
  epochs= list(range(1,len(train_loss)+1))

  plt.figure(figsize=(12,6))

  plt.subplot(1,2,1)
  plt.plot(epochs,train_loss,label="Train Loss")
  plt.plot(epochs,val_loss,label="Val loss")
  plt.title('Loss over epochs')
  plt.xlabel('Epoch')
  plt.ylabel("Loss")
  plt.legend()
  plt.grid(True)

  plt.subplot(1,2,2)
  plt.plot(epochs,train_acc,label="Train Accuracy")
  plt.plot(epochs,val_acc,label="Val accuracy")
  plt.title('Accuracy over epochs')
  plt.xlabel('Epoch')
  plt.ylabel("Accuracy")
  plt.legend()
  plt.grid(True)

  plt.subplot(1,2,3)
  plt.plot(epochs,lrs,label="learning_rate")
  plt.title('Learning rates over epochs')
  plt.xlabel('epoch')
  plt.ylabel('Learning rate')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.savefig('Training.png')
  plt.show()

 

  

##############################################################################
##############################################################################
##############################################################################

def evaluate_model(model,test_loader,device):
  """
  - function to evaluate the model
  Args:
     - model: model to evaluate
     - test_loader(DataLoader): test data
     - device(string): cuda if available
  """
  model.eval()
  all_preds=[]
  all_labels=[]
  with torch.no_grad():
    correct=0
    total=0
    for images,labels in test_loader:
      images= images.to(device)
      labels= labels.to(device)

      outputs= model(images)
      _,pred= torch.max(outputs.data,1)
      all_preds.extend(pred.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

      total+=labels.size(0)
      correct+=(pred==labels).sum().item()
    accuracy= correct/total
    print(f'test accuracy: {accuracy:.4f}')

  #classification report
  from sklearn.metrics import classification_report, confusion_matrix
  print('\n Classification Report: ')
  print(classification_report(all_labels,all_preds,
                              target_names=model.class_names))
  print('\n confusion Matrix:')
  print(confusion_matrix(all_labels,all_preds))

  return accuracy

##############################################################################
##############################################################################
##############################################################################

def predict_image(model,image_path,device,threshold=.9):

  """
   - function to predict an image 
   Args:
    - model: model for inference
    - image_path(string): path to find the image
    - device(String): cuda or cpu
    -threshold(float): decision region
  """
  transform= transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
  ])
  image= Image.open(image_path).convert('RGB')
  tensor= transform(image).unsqueeze(0).to(device=device)
  with torch.no_grad():
    outputs= model(tensor)
    probabilities= F.softmax(outputs,dim=1)
    confidence, pred_idx= torch.max(probabilities,1)
  class_name= model.class_names[pred_idx.item()]
  confidence= confidence.item()

  if confidence <threshold:
    return {'status': 'No highlight','confidence':confidence}
  if class_name in ['Left','Right','Center']:
    return {'status': 'No highlight','reason':class_name,
            'confidence':confidence}
  if class_name =='Card':
    return {'status':'Card','confidence':confidence,'image':image }
  return {'status': 'Highlight','event':class_name,'confidence':confidence}


      
