
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader, random_split
import torch.optim as optim
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

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
    self.backbone= efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    #adapt the classifier for 9 outputs
    num_features= self.backbone.classifier[1].in_features

    #classes in sorted order
    self.class_names = [
        'Cards',
        'Center',
        'Corner',
        'Free-Kick',
        'Left',
        'Penalty',
        'Right',
        'Tackle',
        'To-Subtitue']

    self.class_names_dict = {key: i for i, key in enumerate(self.class_names)}

    self.backbone.classifier= nn.Sequential(
        # first dense Layer
        nn.Linear(num_features,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(.5),

        #second dense Layer
        nn.Linear(512,512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(.5),

        #final classification layer
        nn.Linear(512,len(self.class_names))
    )
  def forward(self,x):
    return self.backbone(x)

##############################################################################
##############################################################################
##############################################################################


def train_model(model,batch_size,num_epochs, learning_rate,device,
                full_dataset):
  """
  -this function is to train the model it takes the following arguments
  Args:
    - model(callable): model to train
    -batch_size(int): the batch size 
    - num_epochs(int): the number of epochs
    -learning_rate(float): learning rate
    - device(string): cuda or cpu
    - full_dataset(Dataset): train_valid dataset
  """

  train_size=int(.8 * len(full_dataset))
  val_size= len(full_dataset)-train_size
  train_dataset, valid_dataset= random_split(full_dataset,[train_size,val_size])

  #Loaders
  train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  valid_loader= DataLoader(valid_dataset, batch_size=batch_size)

  #model setup
  criterion= nn.CrossEntropyLoss()
  optimizer= optim.Adam(model.parameters(),lr=learning_rate)

  #Training metrics storage
  train_loss_history=[]
  val_loss_history=[]
  accuracy_history=[]

  #Training loop
  for epoch in range(num_epochs):
    model.train()
    running_loss=.0

    for images, labels in train_loader:
      images= images.to(device=device)
      labels= labels.to(device=device)
      optimizer.zero_grad()
      outputs= model(images)
      loss=criterion(outputs,labels)
      loss.backward()
      optimizer.step()

      running_loss+=loss.item()*images.size(0)
    #validation phase
    model.eval()
    val_loss=.0
    correct=0
    total=0
    with torch.no_grad():
      for images, labels in valid_loader:
        images= images.to(device=device)
        labels= labels.to(device=device)
        optimizer.zero_grad()
        outputs= model(images)
        loss=criterion(outputs,labels)
        val_loss+=loss.item()*images.size(0)

        _,pred= torch.max(outputs.data,1)
        total +=labels.size(0)
        correct +=(pred==labels).sum().item()

    #calculate metrics
    epoch_train_loss= running_loss/len(train_dataset)
    epoch_val_loss= val_loss/len(valid_dataset)
    epoch_acc= correct/total

    train_loss_history.append(epoch_train_loss)
    val_loss_history.append(epoch_val_loss)
    accuracy_history.append(epoch_acc)

    print(f"{epoch+1}/{num_epochs}")
    print(f'Train Loss:{epoch_train_loss:.4f} | Valid Loss {epoch_val_loss:.4f}')
    print(f'Accuracy:{epoch_acc:.4f}')
  #save model
  torch.save(model.state_dict(),'/content/drive/MyDrive/FootballAnalysis/model_event_classifier.pth')

  #plot training history

  plt.figure(figsize=(12,5))
  plt.subplot(1,2,1)
  plt.plot(train_loss_history,label='Train Loss')
  plt.plot(val_loss_history,label='Val loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(accuracy_history,label='accuracy')
  plt.title('Validation accuracy')
  plt.legend()
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


      
