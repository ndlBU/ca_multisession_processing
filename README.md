# MSP
**M**ulti **S**ession **P**rocessing of calcium activity 

A Python package that allows the cross registration of neuronal ROIs from multiple calcium imaging sessions.

This package is used in conjunction with calcium activity extraction tools to track neuronal activity across multiple sessions. 

## Getting started

A template is provided in demo_pipeline_template.py which contains step-by-step instructions.

### Overview 
1. Get an initial estimates of neuron ROIs from multiple recordings using your favorite tool.
2. Use MSP to find the unique set of neurons.
3. Use the set of identified unique ROIs to update calcium activity estimates.