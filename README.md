# DENN
Experiments for the paper _DENN_, _Handling Black Swan Events in Deep Learning with Diversely Extrapolated Neural Networks_

# Datasets
We are looking for a solution to store data online.
In the meantime, the procedure to generate the data frames from the experiment on imitation learning can be followed using the paper and these links:
- [Repository #1](https://github.com/sanjaythakur/Uncertainty-aware-Imitation-Learning-on-Multiple-Tasks-using-Bayesian-Neural-Networks/tree/master/MuJoCo)
- [Repository #2](https://github.com/sanjaythakur/Multiple_Task_MuJoCo_Domains)
- [Repository #3](https://github.com/sanjaythakur/trpo)

Feel free to contact me if you want me to send the .h5 versions of the datasets.

# Regression
- The ``.sh`` files to train the different models and generate the figuresare available in ``exec/``.

# Classification
- The ``.sh`` files to train the reference function, the deep ensemble and DENN are available in ``exec/``.
- The figures can be recreated using the ``gen_figs.py`` file. Type ``python gen_figs.py --help`` to get more information about the arguments to pass.

# Imitation learning
- Train a PPO agent on Reacher-v1 using the code and default parameters available at https://github.com/sanjaythakur/trpo
  The trained agent is saved in folder saved_models/
  ``./train.py Reacher-v1 -n 60000 -b 50``
- To create the datasets, you first need to modify the settings of MuJoCo following the instructions given in the Repository #2 above.  Then, create trajectories datasets (a graphic interface is required (ssh is NOT enough)): launch: 
  ``bash exec/create_XXX_XXX_dataset.sh``
- To train a reference function on the chosen dataset, launch:
  ``bash exec/train_reference.sh``
  . Note: modify the dataset to use in the ``.sh`` file.
- To train the ensemble, launch: 
  ``bash exec/train_ensemble.sh``
  . Note: modify the dataset to use in the ``.sh`` file.
- To train DENN, launch: 
  ``bash exec/train_denn.sh``
  . Note: modify the datasets to use in the ``.sh`` file.
- To compute statistics and generate the figures, use the ``.sh`` files in ``exec/``. Use the ``--final`` flag to compute stats on the test set rather than the validation set. 
