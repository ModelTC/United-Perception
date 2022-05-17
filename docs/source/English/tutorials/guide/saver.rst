Saving config
=============

The config of saving models and logs.

.. warning::

   * auto_resume: automatically download the latest checkpoint from save_dir when auto_resume is True.
   * Priority: 'auto_resume' > 'opts' > 'resume_model' > 'pretrain_model'.

base
----

Models are automatically saved in the 'save_dir'.

.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints # dir to save checkpoints
        pretrain_model: # pretrain_path 
        results_dir: results_dir  # dir to save detection results. i.e., bboxes, masks, keypoints
        auto_resume: True
