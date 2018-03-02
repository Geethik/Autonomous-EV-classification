#!/usr/bin/env python2.7
import audioBasicIO
import audioFeatureExtraction
import matplotlib.pyplot as plt
[Fs_x, x] = audioBasicIO.readAudioFile("emer/1.wav");
x = audioBasicIO.stereo2mono(x) 
F_x = audioFeatureExtraction.stFeatureExtraction(x, Fs_x, 0.050*Fs_x, 0.025*Fs_x);

[Fs_y, y] = audioBasicIO.readAudioFile("nonemer/9.wav");
y = audioBasicIO.stereo2mono(y) 
F_y = audioFeatureExtraction.stFeatureExtraction(y, Fs_y, 0.050*Fs_y, 0.025*Fs_y);

plt.subplot(2,1,1); plt.plot(F_x[0,:]); plt.xlabel('emer'); plt.ylabel('ZCR'); 
plt.subplot(2,1,2); plt.plot(F_y[0,:]); plt.xlabel('nonemer'); plt.ylabel('ZCR'); plt.show()

plt.subplot(2,1,1); plt.plot(F_x[1,:]); plt.xlabel('emer'); plt.ylabel('Energy'); 
plt.subplot(2,1,2); plt.plot(F_y[1,:]); plt.xlabel('nonemer'); plt.ylabel('Energy'); plt.show()
