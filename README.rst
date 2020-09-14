Link to the Challenge: `Trajnet++ Challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_

Starter Guide (NEW): `Introducing Trajnet++ Framework <https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/>`_

Data Setup
==========

Data Directory Setup
--------------------

All Datasets are stored in DATA_BLOCK

All Models after training are stored in OUTPUT_BLOCK: ``mkdir OUTPUT_BLOCK``

Data Conversion
---------------

For data conversion, refer to trajnetplusplusdataset.

After conversion, copy the converted dataset to DATA_BLOCK

Training LSTMs
==============

python -m trajnetbaselines.lstm.trainer_update 

Three lstm models will be trained by running this script :
Model two: "use_goals_pred_goals" : 
    train the pred_goals model with goals and use the pred_goals to train the pred_trajs model
Model one: "not_use_goals" : 
    train the pred_trajs model without using goals but zero Tensor instead
Model zero: "use_goals_pred_trajs": 
    train the pred_trajs model with the true goal coordinates
    
The training loss of pred_trajs model of these three models is shown in the training_loss.pdf

Citation
========

If you find this code useful in your research then please cite

.. code-block::

    @inproceedings{Kothari2020HumanTF,
      title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
      author={Parth Kothari and Sven Kreiss and Alexandre Alahi},
      year={2020}
    }

