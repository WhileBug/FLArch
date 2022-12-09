# FLArch: A Federated Learning Architecture Supporting Adding Different Modules

FLArch is a experimental framework for FL researchers to quickly experiment on FL.

## Client

- **Weight update pruning**: support model pruning for weight update to accelerate the network transmission.
- **Data poisoning attack**: support simulation of data poisoning attack in clients 
- **Model poisoning attack**: support simulation of model poinsoning attack in clients
- **Personalization of local model**: support local model personalization, optimization

## Server

- **Participant selection**: support adding different participant selection modules
- **Aggregation algorithm**: support adding different aggregation algorithms at server
- **Updates history preserve**: support preserve the history for each model update
- **Byzantine client detection**: support Byzantine attacker detection process

## Profile

- **Network latency simulation**: support simulate the network latency to profile network performance
- **Reputation management**: support reputation management for different clients