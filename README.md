## Source Code of Interactive System-wise Anomaly Detection


### Abstract

Anomaly detection, where data instances are discovered containing feature patterns different from the majority, plays a fundamental role in various applications. 
However, it is challenging for existing methods to handle the scenarios where the instances are systems whose characteristics are not readily observed as data. 
Appropriate interactions are needed to interact with the systems and identify those with abnormal responses. 
Detecting system-wise anomalies is a challenging task due to several reasons including: 

- how to formally define the system-wise anomaly detection problem?
- how to find the effective activation signal for interacting with systems to progressively collect the data and learn the detector?
- how to guarantee stable training in such a non-stationary scenario with real-time interactions?
To address the challenges, we propose InterSAD (Interactive System-wise Anomaly Detection)?

Specifically, first we adopt Markov decision process to model the interactive systems, and define anomalous systems as anomalous transition and anomalous reward systems.
Then, we develop an end-to-end approach which includes an encoder-decoder module that learns system embeddings, and a policy network to generate effective activation for separating embeddings of normal and anomaly systems. 
Finally, we design a training method to stabilize the learning process, which includes a replay buffer to store historical interaction data and allow them to be re-sampled.
Experiments on two benchmark environments, including identifying the anomalous robotic systems and detecting user data poisoning in recommendation models, demonstrate the superiority of InterSAD compared with baselines methods.


### Dependency:
````angular2html
scikit-learn>=0.23.1
torch>=1.7.0 
gym>=0.17.3
pybullet>=3.0.6
matplotlib>=2.2.3
seaborn>=0.9.0
````

### Reproduce our results:

To train EPG/R on *VirtualTaobao*, run
````angular2html
cd InterSADR
python InterSADR_train.py --exp virtual_taobao
cd ../
````

To reproduce Table 3, run 
````angular2html
cd InterSADR
python InterSADR_test_auc.py
cd ../
````

To reproduce Figure 5(a), run
````angular2html
cd InterSADR
python auc_vs_var_plot.py
cd ../
````

To reproduce Figure 5(b), run
````angular2html
cd InterSADR
python InterSADR_scatter.py
cd ../
````

To reproduce Figure 5(c), run
````angular2html
cd InterSADR
python RR_scatter.py
cd ../
````

To train InterSAD-T on *HalfCheetah*, run
````angular2html
cd InterSADT
python InterSADT_noSTU.py --exp halfcheetah
cd ../
````

To reproduce Table 2, run
````angular2html
cd InterSADT
python InterSADT_test_auc.py 
cd ../
````

To reproduce Figures 6 (b), open auc_plot.py, recover lines 22-29, and run
````angular2html
cd InterSADT
python auc_plot.py
cd ../
````

To reproduce Figures 6 (c), open auc_plot.py, recover lines 31-38, and run
````angular2html
cd InterSADT
python auc_plot.py
cd ../
````

To reproduce Figures 7 (a), run 
````angular2html
cd InterSADT
python auc_plot_std_noise.py
cd ../
````

To reproduce Figures 7 (b), run 
````angular2html
cd InterSADT
python auc_plot.py
cd ../
````

To reproduce Figure 7 (c), run
````angular2html
cd InterSADT
python auc_plot_trajectory_length.py
cd ../
````

### Device Information of Developer:

| Device attribute | Value |
| ---------------- | ----- |
| Computing infrastructure | CPU |
| CPU model | Intel Core i5 |
| Basic frequency | 1.4GHz |
| Memory | 8GB |
| Operating system | Linux |
| InterSAD-T training efficiency | 0.5 CPU sec/iteration  |
| InterSAD-R training efficiency | 0.5 CPU sec/iteration  |
