## Source Code of Towards Anomaly Detection in Markov Decision Process


### Abstract
Anomalous patterns widely exist in real-world Markov decision process (MDP), such as users' malicious clicks in a recommender system or unpredictable responses in controlling systems.
Motivated by this, we study anomaly detection in MDP systems in this work, which is a challenging problem because (i) existing work lacks formal definitions for anomalous patterns in MDP;
(ii) the detector needs to interact with the environment to progressively collect the data, which contradicts the assumptions of many traditional anomaly detectors that operate on static datasets; (iii) the state and action spaces of real-world MDPs can be continuous and noisy with high-dimension, which poses challenges in the efficiency of the algorithms. 
To tackle these challenges, we formalize the anomalous rewards and state transition in MDP, and propose Equilibrious Policy Gradient (EPG) for end-to-end anomaly detection in MDP. 
In EPG, the policy minimizes the pair-wise distances of normal system representations to encourage consistent trajectories in normal MDP systems and isolate anomalous ones. 
Experiments on two benchmark environments, including user attack detection in recommender systems and anomalous robotic system detection, demonstrate the superiority of EPG compared with baselines.


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

To train EPG/T on *VirtualTaobao*, run
````angular2html
cd EPGR
python EPGR_train.py --exp virtual_taobao
cd ../
````

To reproduce Table 1, run 
````angular2html
cd EPGR
python EPGR_test_auc.py
cd ../
````

To reproduce Figure 4(a), run
````angular2html
cd EPGR
python EPGR_scatter.py
cd ../
````

To reproduce Figure 4(b), run
````angular2html
cd EPGR
python RR_scatter.py
cd ../
````

To train EPG/T on *HalfCheetah*, run
````angular2html
cd EPGT
python EPGT_train.py --exp halfcheetah
cd ../
````

To reproduce Table 2, run
````angular2html
cd EPGT
python EPGT_test_auc.py 
cd ../
````

To reproduce Figures 5 (a) and (b), run 
````angular2html
cd EPGT
python auc_plot.py
cd ../
````

To reproduce Figure 6 (a), run
````angular2html
cd EPGT
python auc_plot_sampling_number.py
cd ../
````

To reproduce Figure 6 (b), run
````angular2html
cd EPGT
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
| EPG/R training time | 0.5 CPU sec/step×1000 steps  |
| EPG/T training time | 0.5 CPU sec/step×2000 steps  ！