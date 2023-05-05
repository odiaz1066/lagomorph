# Lagomorph

Reinforcement Learning framework focused on quick prototyping of DQN models. Based on CleanRL, Pytorch, Gymnasium and Unity ML Agents.
It's based around the idea of modifying the train files directly, making one for each kind of experiment.
It's basically a stripped down fork of CleanRL, leaving only those files related to DQN.

## Getting Started

This framework will be created for execution in both Google Colab and Azure Machine Learning, but it should work on any Python <=3.10 environment.

## Prerequisites

Python <= 3.10

To train on Atari environments, the ffmpeg package is needed. It can be installed by executing (May require sudo permissions):
```
apt-get update
apt-get install ffmpeg
```
The libgl1 package may also be used instead if you want to save on ~200 MB of storage.

```
Examples
```

## Deployment

Add additional notes about how to deploy this on a production system.

## Resources

Add links to external resources for this project, such as CI server, bug tracker, etc.
