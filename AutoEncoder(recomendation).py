#Treinamento e avaliação de AutoEncoder(pytorch) para recomendação de filmes

#Importações
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel as parallel


#Importação dos Dados
movies = pd.read_csv("ml-1m\movies.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')


users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


ratings= pd.read_csv("ml-1m/ratings.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')


training_set = pd.read_csv("ml-1m/training_set.csv", delimiter = ',')

test_set = pd.read_csv("ml-1m/test_set.csv" , delimiter = ',')

#Transformação dos Dados
training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int')


nb_users = int(max (max(training_set[:,0]), max(test_set[:,0])))

nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

#Conversão dos Dados em formato aceito pelo torch
def convert(data):
    new_data = []
    for id_users in range(1, nb_users +1 ):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
        
    return new_data


training_set = convert(training_set)

test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)

test_set = torch.FloatTensor(test_set)

#Criação da Classe com a estrutura do auto_encoder
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 50)
        self.fc2 = nn.Linear(50,40)
        self.fc3 = nn.Linear(40,50)
        self.fc4 = nn.Linear(50, nb_movies)
        self.activation = nn.Sigmoid()
        
    def forward(self,x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
        
sae = SAE() #Instancia a classe
criterion = nn.MSELoss() #Função de Loss
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.4) #Otimizador
nb_epochs = 200 #Epochs

#Treinamento do Auto Encoder
for epochs in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_users in range(nb_users):
        input = Variable(training_set[id_users]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.required_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10 )
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.0
            optimizer.step()
    print('Epoch: ' + str(epochs)+ '\t Loss: ' + str(train_loss/s))
    

#Avaliação do Modelo
test_loss = 0
s = 0.
for id_users in range(nb_users):
    input = Variable(training_set[id_users]).unsqueeze(0)
    target = Variable(test_set[id_users]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.required_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        loss.backward()
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print("Loss: " + str(test_loss/s))

    
    
    

    




        
    
