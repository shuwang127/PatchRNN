# PatchClassificationRNN

```shell script
pip install clang
``` 
    # tensor data processing.
    xTrain = torch.from_numpy(dTrain).long().cuda()
    yTrain = torch.from_numpy(lTrain).long().cuda()
    xValid = torch.from_numpy(dTest).long().cuda()
    yValid = torch.from_numpy(lTest).long().cuda()
    # batch size processing.
    train = torchdata.TensorDataset(xTrain, yTrain)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)

    # build the model of recurrent neural network.
    preWeights = torch.from_numpy(preWeights)
    model = TextRNN(preWeights, hiddenSize=hiddenSize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('[Demo] --- RNNType: TextRNN | HiddenNodes: %d  ---' % (hiddenSize))
    print('[Para] BatchSize=%d, LearningRate=%.4f, MaxEpoch=%d, PerEpoch=%d.' % (batchsize, learnRate, maxEpoch, perEpoch))
    # optimizing with stochastic gradient descent.
    optimizer = optim.Adam(model.parameters(), lr=learnRate)
    # seting loss function as mean squared error.
    criterion = nn.CrossEntropyLoss()
    # memory
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # run on each epoch.
    accList = [0]
    for epoch in range(maxEpoch):
        # training phase.
        model.train()
        lossTrain = 0
        predictions = []
        labels = []
        for iter, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.contiguous().view(-1)
            label = label.to(device)
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(yhat, label)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item() * len(label)
            preds = yhat.max(1)[1]
            predictions.extend(preds.int().tolist())
            labels.extend(label.int().tolist())
            torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        lossTrain /= len(dTrain)
        # train accuracy.
        accTrain = accuracy_score(labels, predictions) * 100

        # validation phase.
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                data = data.to(device)
                label = label.contiguous().view(-1)
                label = label.to(device)
                yhat = model.forward(data)  # get output
                # statistic
                preds = yhat.max(1)[1]
                predictions.extend(preds.int().tolist())
                labels.extend(label.int().tolist())
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        # valid accuracy.
        accValid = accuracy_score(labels, predictions) * 100
        accList.append(accValid)

        # output information.
        if 0 == (epoch + 1) % perEpoch:
            print('[Epoch %03d] loss: %.3f, train acc: %.3f%%, valid acc: %.3f%%.' % (epoch + 1, lossTrain, accTrain, accValid))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tempPath + '/model_TextRNN.pth')
        # stop judgement.
        if (epoch + 1) >= perEpoch and accList[-1] < min(accList[-perEpoch:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tempPath + '/model_TextRNN.pth'))
    print('[Info] Text Recurrent Neural Network classifier training done!')
    return model