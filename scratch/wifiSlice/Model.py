import torch
import torch.nn as nn
import numpy as np

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
    Trains the model and returns actions, targets for a given state
    '''
    def __init__(self, device):
        self.device = device
        self.model = BasicModel().to(self.device)
        self.optim = torch.optim.Adam(lr=10e-4, params=self.model.parameters())
    
    def getActionTuple(self, obs, action):
        #choose best action
        #explore

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
        increments or decrements action values based on the action tuple
        0 --> -1
        1 --> do nothing
        2 --> +1

        also if value is min no decrement is possible and
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
        obs --> action --> obs_new
        '''

        self.model.train()
        featA, featB, featC = self.getInputFeaturesFromObservation(obs)
        actA, actB, actC = torch.unbind(self.convertActionToTensor(action))
        
        featA = torch.cat([featA, actA])
        featB = torch.cat([featB, actB])
        featC = torch.cat([featC, actC])

        x = torch.cat([featA, featB, featC]).unsqueeze(0).to(self.device)
        prediction = self.model(x)

        #index using action tuple
        predicted_target = prediction[0][action_tuple]
        real_target = self.getTarget(obs_new, action).to(self.device)
        loss = (predicted_target - real_target).pow(2).mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def getTarget(self, obs, action):
        '''
        returns 
        - Latency, done
        - Error probability (tx-rx)/tx
        - Transmission power
        - Spectral efficiency (sum(rxpackets)/time)/bandwidth
        '''

        obsTensors = self.convertObsToTensors(obs)
        
        mean_latencyA = torch.mean(obsTensors[0][3])
        mean_error_probA = torch.mean(1 - obsTensors[0][2]/obsTensors[0][1])

        txPowerB = action["txPower"][1]     #Todo : multiply with numStationsB?
        
        throughput = sum([sum(obsTensors[i][2]) * 1472 * 8 for i in range(3)])/1000000.0  #1472 is payload size, 8 bits
        #Todo : Divide by sim time ?
        totalBandWidth = sum([chWidthFromChNum(action["chNum"][i]) for i in range(3)])
        se = throughput/totalBandWidth

        return mean_latencyA + mean_error_probA + txPowerB + se

    def getInputFeaturesFromObservation(self, obs):
        obsTensors = self.convertObsToTensors(obs)
        funcs = [torch.mean, torch.var, torch.min, torch.max]
        featuresA = torch.stack([func(elem) for elem in obsTensors[0] for func in funcs])
        featuresB = torch.stack([func(elem) for elem in obsTensors[1] for func in funcs])
        featuresC = torch.stack([func(elem) for elem in obsTensors[2] for func in funcs])

        #print(featuresA.shape)  #(16-d vector)
        return featuresA, featuresB, featuresC

    def convertObsToTensors(self, obs):
        drA = torch.Tensor(obs["SliceA"][0]).float()      #datarate
        tpA = torch.Tensor(obs["SliceA"][1]).float()      #txpackets
        rpA = torch.Tensor(obs["SliceA"][2]).float()      #rxpackets
        lA = torch.Tensor(obs["SliceA"][3]).float()       #latency

        drB = torch.Tensor(obs["SliceB"][0]).float()
        tpB = torch.Tensor(obs["SliceB"][1]).float()
        rpB = torch.Tensor(obs["SliceB"][2]).float()
        lB = torch.Tensor(obs["SliceB"][3]).float()

        drC = torch.Tensor(obs["SliceC"][0]).float()
        tpC = torch.Tensor(obs["SliceC"][1]).float()
        rpC = torch.Tensor(obs["SliceC"][2]).float()
        lC = torch.Tensor(obs["SliceC"][3]).float()
        return [(drA, tpA, rpA, lA),
                (drB, tpB, rpB, lB),
                (drC, tpC, rpC, lC)]

    def convertActionToTensor(self, action):
        chNum = torch.Tensor(action["chNum"]).float()
        gi = torch.Tensor(action["gi"]).float()
        mcs = torch.Tensor(action["mcs"]).float()
        txPower = torch.Tensor(action["txPower"]).float()

        return torch.stack([chNum, gi, mcs, txPower]).transpose(0, 1)

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        input_size = 60
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

