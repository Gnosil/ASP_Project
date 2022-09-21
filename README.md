# ASP_Project
This project is for Audio signal class final project.

## VAD part
The current VAD folder contains a pipline for TDNN-LSTM frame work and a Pyannote framework using AMI dataset which is not the main problem we are solving in this project. The pipeline here is just for comparing the result with our result. In order to reproduce the result of these pipelines, please first install the OVAD package using the following commands:
> git clone https://github.com/desh2608/ovad.git
> python setup.py install

## Goal of this project
In this project we are trying to work on the Overlapped Voice Activity Detection problem as a Multiclass classification problem with three different classes: {Silence, Speech, Overlapped speech}. We are trying to work on this problem in a signal processing aspect. We will try to solve this problem using Energy Based, Format detector, and Pitch detector. We will also try to work on different speech augmentation methods to provide a better result.
