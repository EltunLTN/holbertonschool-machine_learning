#!/usr/bin/env python3
"""Deep Neural Network - PyTorch versiyası (eyni məntiq, hazır kitabxana ilə)"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class DeepNeuralNetwork(nn.Module):
    def __init__(self, nx, layers, activation='sig'):
        super().__init__()

        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        act_layer = nn.Sigmoid() if activation == 'sig' else nn.Tanh()

        sizes = [nx] + layers
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(sizes[i], sizes[i + 1]))
            modules.append(act_layer)
        modules.append(nn.Linear(sizes[-2], sizes[-1]))   # çıxış qatı (logits)
        # Softmax ƏLAVƏ EDİLMİR -- CrossEntropyLoss training zamanı
        # onu daxilində özü tətbiq edir; inference-də əl ilə softmax edəcəyik

        self.net = nn.Sequential(*modules)
        self.activation = activation

    def forward(self, X):
        return self.net(X)

    def train_model(self, X, Y, iterations=5000, alpha=0.05,
                     verbose=True, graph=True, step=100):
        X_t = torch.tensor(X, dtype=torch.float32).T   # (m, nx)
        Y_t = torch.tensor(Y, dtype=torch.float32).T   # (m, classes)

        optimizer = optim.SGD(self.parameters(), lr=alpha)
        criterion = nn.CrossEntropyLoss()

        costs, steps = [], []

        for i in range(iterations + 1):
            output = self(X_t).T   # (classes, m) formatına qaytarmaq
            loss = criterion(output.T, Y_t.argmax(dim=1))

            if i % step == 0 or i == iterations:
                costs.append(loss.item())
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {loss.item()}")

            if i < iterations:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        with torch.no_grad():
            logits = self(X_t).T
            A = torch.softmax(logits, dim=0)
            pred = torch.zeros_like(A)
            pred[A.argmax(dim=0), torch.arange(A.shape[1])] = 1
        return pred.numpy(), loss.item()

    def save(self, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        torch.save(self, filename)

    @staticmethod
    def load(filename):
        import os
        if not os.path.exists(filename):
            return None
        return torch.load(filename, weights_only=False)
    