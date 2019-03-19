Fig. allocated_resource_regression

To estimate the allocated CPU resource more accurately, we record the CPU resouces that the task occupies under different wordload on each devices and draw the Fig. X. It is observed that the record points that are collected in the same device (which are labelled with the same color) is approximately located on a straight line. This important fact inspires the employment of the linear regression model, which trains a linear polynomial to minimize the error between the prediction and record points. The trained linear regression models for each device have been drawn as lines in Fig. X, with the same color of the record points on the same device. As Fig. X shows, all the devices draws a nagative regression model except for the server. This implies that for most devices the device allocate less CPU resource for the task as the workload increases, but for the resource-abundant server, the allocated CPU resource keeps in a stable level regradless of the change of workload. 

Remarkably,  Fig. X presents another essential fact that the record points separate clearly in three different levels, which corresponses to their computing capability classes. The device with higher computing capability allocates less CPU resource for the task and draws a regression line with smaller absoluate value of slope.



Fig. energy_curve

We also record the real-time energy consumption of the sending process on both the two stages. Fig. X shows the typical real-time energy consumption of the whole communication process in the client (Raspberry Pi). To better identify the key image sending process in porbing stage and transfer stage, we set up three sleep timers (sleep for 2 s, 4 s and 2 s respectively) during the whole process, which is labelled in the shadow areas.  The two peaks marked in Fig. X represents the moments that sending packets from the client to the  server, where the first is in the probing stage and the other is in the transfer stage.  The record energy power of the two peaks are 175 mW and 242 mW respectively, while in our repeated experiments the average energy power of the sending process on the two stages are 165 mW and 250 mW respectively.

