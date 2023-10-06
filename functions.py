import pandas as pd
import torch
import torch.nn.functional as F
import time

# Qui raccogliamo alcune funzioni utili!

# Funzione per addestrare un modello neurale di classificazione
def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu",to_print=True):
    
    lista_epoch = []
    lista_train_loss = []
    lista_val_loss = []
    lista_accuracy = []
    #lista_lr = []
    lista_time = []
    
    print("Begin training...")
    #scheduler = ExponentialLR(optimizer, gamma=0.8)
    
    for epoch in range(1, epochs+1):
        start = time.process_time()
        lista_epoch.append(epoch)
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad() # clear gradients for next train
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            # print(output, output.shape)
            # print(targets, targets.shape)
            loss = loss_fn(output, targets)
            loss.backward() # backpropagation, compute gradients
            optimizer.step() # apply gradients
            training_loss += loss.data.item() * inputs.size(0)
            # print(training_loss,loss.data.item(),inputs.size(0))
        
            if to_print:
                break
        training_loss /= len(train_loader.dataset)
        lista_train_loss.append(format(training_loss, ".6f"))
        
        if to_print:
            break
        with torch.no_grad():
            model.eval()
            num_correct = 0 
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                output = model(inputs)
                targets = targets.to(device)
                
                loss = loss_fn(output,targets) 
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)
            lista_val_loss.append(format(valid_loss, ".6f"))
            lista_accuracy.append(num_correct/num_examples)
            
        #scheduler.step()
        #lista_lr.append((format(scheduler.get_last_lr()[-1], ".6f")))    
        end = time.process_time()
        duration = end-start
        lista_time.append(duration)

        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f},  Time: {:.6f}'.format(epoch, training_loss,
        valid_loss, num_correct / num_examples, duration))
    if to_print:
        return 
    else:
        dataframe = pd.DataFrame({"Epoch": lista_epoch, "Training Loss":lista_train_loss, "Validation Loss":lista_val_loss, "Accuracy":lista_accuracy,  "Time":lista_time})
        return dataframe