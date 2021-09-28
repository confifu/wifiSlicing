import torch
import torch.nn as nn
import numpy as np
from random import randint as ri

def multiDimArgmax(arr):
    #TODO : Make it work for batch
    a = arr.cpu().numpy()
    indices = np.unravel_index(np.argmax(a, axis=None), a.shape)
    return indices

def chWidthFromChNum(chNum):
    if chNum < 29: return 20
    elif chNum < 43: return 40
    elif chNum < 50: return 80
    else : return 160

class ModelHelper():
    '''
    Helper class that trains the model and returns actions, targets for a given state
    '''
    def __init__(self, device, outdir, resume_from, epsilon = 0.1):
        self.device = device
        self.model = BasicModel().to(self.device)
        self.optim = torch.optim.Adam(lr=10e-4, params=self.model.parameters())
        self.outdir = outdir
        self.prevEpoch = -1
        self.trainLosses = []
        self.epsilon = epsilon
        if resume_from is not None:
            checkpoint = torch.load(resume_from)
            self.prevEpoch = checkpoint['epoch']

            print("Resuming from checkpoint at, ", resume_from, ", epoch, ", self.prevEpoch)

            self.trainLosses = checkpoint['trainLosses']
            self.model.load_state_dict(checkpoint['state_dict'], strict = True)
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

            del checkpoint

    def getActionTuple(self, obs, action):
        '''
        Returns a vector with values {0,1,2} using the trained model and
        last observation and action, The vector decides whether a variable
        should be decreased, increased or remain constant.
        '''

        if ri(0, 10)/10 > self.epsilon:
            return tuple([ri(0, 2) for i in range(12)])

        with torch.no_grad():
            self.model.train(False)
            featA, featB, featC = self.getInputFeaturesFromObservation(obs)
            actA, actB, actC = torch.unbind(self.convertActionToTensor(action))

            featA = torch.cat([featA, actA])
            featB = torch.cat([featB, actB])
            featC = torch.cat([featC, actC])

            x = torch.cat([featA, featB, featC]).unsqueeze(0).to(self.device)
            predicted_target = self.model(x)

            indices = multiDimArgmax(predicted_target[0])
            return indices

    def getActionFromActionTuple(self, action_tuple, action):
        '''
        Returns a new action(that can be put into simulations) from action tuple
        increments or decrements action values based on the action tuple
        0 --> -1
        1 --> do nothing
        2 --> +1

        if value is min no decrement is possible and
        if value is max no increment is possible
        '''
        action_names = ["chNum", "gi", "mcs", "txPower"]
        max_min_dict = {
            "chNum":[0, 52],
            "gi":[0, 2],
            "mcs" : [0, 11],
            "txPower" : [1, 20]
        }
        for j, action_name in enumerate(action_names):
            for i in range(3):
                if action_tuple[3*j + i] == 0 :
                    action[action_name][i] = max(action[action_name][i] - 1, max_min_dict[action_name][0])
                elif action_tuple[3*j + i] == 2 :
                    action[action_name][i] = min(action[action_name][i] + 1, max_min_dict[action_name][1])
        return action

    def trainModel(self, obs, action, action_tuple, obs_new):
        '''
        Use the new observation to train the model. Also returns the loss of this step.
        obs, action are input to the model. The model predicts the target for all possible
        actions. action_tuple is used to select the predicted target for the best action we
        chose before. Real target value is predicted using obs_new.
        obs --> action --> obs_new
        '''

        self.model.train()
        featA, featB, featC = self.getInputFeaturesFromObservation(obs)
        actA, actB, actC = torch.unbind(self.convertActionToTensor(action))
        
        featA = torch.cat([featA, actA])
        featB = torch.cat([featB, actB])
        featC = torch.cat([featC, actC])
        #Todo : Normalize

        x = torch.cat([featA, featB, featC]).unsqueeze(0).to(self.device)
        prediction = self.model(x)

        #index using action tuple
        predicted_target = prediction[0][action_tuple]
        real_target = self.getTarget(obs_new, action).to(self.device)
        loss = (predicted_target - real_target).pow(2).mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def getTarget(self, obs, action):
        '''
        returns a function of the following
        - Latency
        - Error probability (tx-rx)/tx
        - Transmission power
        - Spectral efficiency (sum(rxpackets)/time)/bandwidth

        We want to maximize the target value.
        '''

        obsTensors = self.convertObsToTensors(obs)
        
        mean_latencyA = torch.mean(obsTensors[0][3])
        mean_error_probA = torch.mean(1 - obsTensors[0][2]/obsTensors[0][1])
        targetA = -mean_latencyA - mean_error_probA                               #eMBB

        mean_latencyB = torch.mean(obsTensors[1][3])
        mean_error_probB = torch.mean(1 - obsTensors[1][2]/obsTensors[1][1])
        txPowerB = action["txPower"][1]     #Todo : multiply with numStationsB?
        targetB = - mean_latencyB - mean_error_probB - txPowerB                   #Minimize power used as well mMTC

        mean_latencyC = torch.mean(obsTensors[2][3])
        mean_error_probC = torch.mean(1 - obsTensors[2][2]/obsTensors[2][1])
        targetC = -10*mean_latencyC - mean_error_probC                               #Focus more on latency URLLC

        throughput = sum([sum(obsTensors[i][2]) * 1472 * 8 for i in range(3)])/1000000.0  #1472 is payload size, 8 bits #Todo : Divide by sim time ?
        totalBandWidth = sum([chWidthFromChNum(action["chNum"][i]) for i in range(3)])
        se = throughput/totalBandWidth

        return targetA + targetB + targetC + se

    def getInputFeaturesFromObservation(self, obs):
        obsTensors = self.convertObsToTensors(obs)
        obsTensors = self.normalizeObs(obsTensors)
        funcs = [torch.mean, torch.var, torch.min, torch.max]
        featuresA = torch.stack([func(elem) for elem in obsTensors[0] for func in funcs])
        featuresB = torch.stack([func(elem) for elem in obsTensors[1] for func in funcs])
        featuresC = torch.stack([func(elem) for elem in obsTensors[2] for func in funcs])

        #print(featuresA.shape)  #(20-d vector)
        return featuresA, featuresB, featuresC

    def convertObsToTensors(self, obs):
        drA = torch.Tensor(obs["SliceA"][0]).float()      #datarate
        tpA = torch.Tensor(obs["SliceA"][1]).float()      #txpackets
        rpA = torch.Tensor(obs["SliceA"][2]).float()      #rxpackets
        lA = torch.Tensor(obs["SliceA"][3]).float()       #latency
        rPowA = torch.Tensor(obs["SliceA"][4]).float()    #rxPower

        drB = torch.Tensor(obs["SliceB"][0]).float()
        tpB = torch.Tensor(obs["SliceB"][1]).float()
        rpB = torch.Tensor(obs["SliceB"][2]).float()
        lB = torch.Tensor(obs["SliceB"][3]).float()
        rPowB = torch.Tensor(obs["SliceB"][4]).float()


        drC = torch.Tensor(obs["SliceC"][0]).float()
        tpC = torch.Tensor(obs["SliceC"][1]).float()
        rpC = torch.Tensor(obs["SliceC"][2]).float()
        lC = torch.Tensor(obs["SliceC"][3]).float()
        rPowC = torch.Tensor(obs["SliceC"][4]).float()

        return [(drA, tpA, rpA, lA, rPowA),
                (drB, tpB, rpB, lB, rPowB),
                (drC, tpC, rpC, lC, rPowC)]

    def normalizeObs(self, obsTensors):
        (drA, tpA, rpA, lA, rPowA),(drB, tpB, rpB, lB, rPowB),(drC, tpC, rpC, lC, rPowC) = obsTensors
        drA = drA / 100
        tpA = tpA / 100000
        rpA = rpA / 100000
        lA = lA / 1000
        rPowA = rPowA / 20

        drB = drB / 100000
        tpB = tpB / 100
        rpB = rpB / 100
        lB = lA / 1000
        rPowB = rPowB / 20

        drC = drC / 100
        tpC = tpC / 100000
        rpC = tpC / 100000
        lC = lC / 1000
        rPowC = rPowC / 20
        return [(drA, tpA, rpA, lA, rPowA),
                (drB, tpB, rpB, lB, rPowB),
                (drC, tpC, rpC, lC, rPowC)]


    def convertActionToTensor(self, action):
        '''
        Convert to tensor and normalize
        '''
        chNum = torch.Tensor(action["chNum"]).float() / 60
        gi = torch.Tensor(action["gi"]).float()
        mcs = torch.Tensor(action["mcs"]).float() / 10
        txPower = torch.Tensor(action["txPower"]).float() / 20

        return torch.stack([chNum, gi, mcs, txPower]).transpose(0, 1)

    def saveModel(self, trainLosses):
        self.prevEpoch = self.prevEpoch + 1
        self.trainLosses.extend(trainLosses)
        checkpoint = {
            'epoch': self.prevEpoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'trainLosses' : self.trainLosses,
        }
        torch.save(checkpoint,
                    self.outdir + "checkpoint" + str(self.prevEpoch) + '.pt')

class BasicModel(nn.Module):
    '''
    Basic model that predicts target value for 3^12 possible actions
    given the previous observation and the action.
    '''
    def __init__(self):
        super(BasicModel, self).__init__()
        input_size = 72
        output_size = pow(3, 12)

        self.layers = nn.Sequential(
            nn.Linear(input_size, pow(2, 7)),
            nn.ReLU(),
            nn.Linear(pow(2, 7), pow(2, 9)),
            nn.ReLU(),
            nn.Linear(pow(2, 9), output_size)
        )
    def forward(self, x):
        x = self.layers(x)
        shape = tuple([-1] + [3 for i in range(12)])
        x = x.reshape(shape)
        return x

