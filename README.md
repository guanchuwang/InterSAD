## Source Code of Interactive System-wise Anomaly Detection


### Abstract

Existing work is incapable to formulate the anomaly detection problem of interactive scenarios where the behaviors of each system could not be explicitly observed as data unless an activation signal is provided, and anomalous systems tend to behave or response differently from others due to the faulted or malicious inner characteristics.
This is a challenging problem due to the fact that:

- Existing work lacks formal definitions for interactive systems-wise anomaly detection;
- It relies on real-time interaction with the systems to progressively collect the data for learning the detector, which contradicts the assumptions of traditional anomaly detection that operate on static datasets; 
- The data collected from real-time interaction with the systems has non-stationary and noisy distribution, which can terribly affect the stability of training procedure.

To address these challenge, we 
- Adopt Markov Decision Process (MDP) to formulate the interactive systems, and formally define \emph{anomalous system} including the anomalous transition and reward systems;
- Propose an Interactive System-wise Anomaly Detection (InterSAD) method which adopts an encoder-decoder to learn the embedding of the systems, and learn a policy to neutralize the system embeddings so that the anomalous systems can be isolated according to inconsistent behaviors;
- Adopt Experience Replay Mechanism (ERM) to stabilize the training process by enqueuing the data collected from the real-time interaction to the replay buffer and resampling the data for training, which encourages more stationary distribution of the training data.

Experiments on two benchmark environments, including identifying the anomalous robotic system and attack detection in recommender systems, demonstrate the superiority of InterSAD compared with baselines.


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
| InterSAD-R training efficiency | 0.5 CPU sec/iteration  |
| InterSAD-T training efficiency | 0.5 CPU sec/iteration  |