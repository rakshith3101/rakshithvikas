{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled4.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyONAeWiuTttFiu1d5iH2Ome",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rakshith3101/rakshithvikas/blob/main/LSTM%20machine%20learning%20model.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3L8sjpxcyJWK"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "import pandas as pd\n",
        "import torch.nn as nn\n",
        "import torch\n",
        "%matplotlib inline\n",
        "x=torch.linspace(0,799,800)\n",
        "y=torch.sin(x*2*3.1416/40)\n",
        "\n",
        "testsize=40\n",
        "train_data=y[:-testsize]\n",
        "test_data=y[-testsize:]\n",
        "plt.figure(figsize=(12,4))\n",
        "plt.xlim(-10,801)\n",
        "plt.xlabel('x')\n",
        "plt.ylabel('sin')\n",
        "plt.title('sin wave cureve')\n",
        "plt.grid(True)\n",
        "plt.plot(train_data.numpy(),color='#8000ff')\n",
        "plt.plot(range(600,640),test_data.numpy(),color='#ff8000')\n",
        "plt.show()\n",
        "\n",
        "def input_data(seq,ws):\n",
        "    out = []\n",
        "    L = len(seq)\n",
        "    \n",
        "    for i in range(L-ws):\n",
        "        window = seq[i:i+ws]\n",
        "        label = seq[i+ws:i+ws+1]\n",
        "        out.append((window,label))\n",
        "    \n",
        "    return out\n",
        "window_sizw=40\n",
        "train_set=input_data(train_data,window_sizw)\n",
        "class LSTM(nn.Module):\n",
        "  def __init__(self,input_size=1,hidden_size=50,out_size=1):\n",
        "    super().__init__()\n",
        "    self.hidden_size=hidden_size\n",
        "    self.lstm=nn.LSTM(input_size,hidden_size)\n",
        "    self.linear=nn.Linear(hidden_size,out_size)\n",
        "    self.hidden=(torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))\n",
        "\n",
        "\n",
        "    def forward(self,seq):\n",
        "        lstm_out, self.hidden = self.lstm(seq.view(len(seq),1,-1), self.hidden)\n",
        "        pred = self.linear(lstm_out.view(len(seq),-1))\n",
        "        return pred[-1]\n",
        "torch.manual_seed(42)\n",
        "model=LSTM()\n",
        "criteron=nn.MSELoss()\n",
        "optimizer=torch.optim.SGD(model.parameters(), lr=0.01)\n",
        "model\n",
        "epochs=10\n",
        "future=40\n",
        "for i in range(epochs):\n",
        "  for seq,y_train in train_data:\n",
        "    optimizer.zero_grad()\n",
        "    model.hidden(torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))\n",
        "    y_pred=model(seq)\n",
        "    loss=optimizer(y_pred,y_train)\n",
        "    loss.backward()\n",
        "    optimizer.step()\n",
        "  print(f\"Epoch {i} Loss: {loss.item()}\")\n",
        "  pred=train_set[-window_sizw:].tolist()\n",
        "  for f in range(future):\n",
        "    seq=torch.FloatTensor(pred[-window_sizw:]) \n",
        "    with torch.no_grad():\n",
        "      model.hidden=(torch.zeros(1,1,model.hidden_size),torch.zeros(1,1,model.hidden_size))\n",
        "      pred.append(modle(seq).item())\n",
        "  loss = criteron(torch.tensor(pred[-window_sizw:]), y[760:])\n",
        "\n",
        "  print(f\"Performance on test range: {loss}\")\n",
        "\n",
        "\n",
        "  plt.figure(figsize=(12,4))\n",
        "  plt.xlim(700,800)\n",
        "  \n",
        "  plt.grid(True)\n",
        "  plt.plot(y.numpy(),color='#8000ff')\n",
        "  plt.plot(range(760,800),preds[window_size:],color='#ff8000')\n",
        "  plt.show()\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}