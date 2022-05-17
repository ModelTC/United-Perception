Environment
===========

Introducing the function which can be realized by changing environment variables.

  .. code-block:: bash

     export xxx=xxxx

up.__init__
-----------

UP can add 'plugins' into environment variables to read its path.

  .. code-block:: python

     _PPPATH = 'PLUGINPATH'
     if _PPPATH not in os.environ:
         return
     path_list = os.environ[_PPPATH].split(':')

up.runner.base_runner
---------------------

UP adds batchsize, display, dataset_timer into environment variables.

* BATCH_SIZE controls batchsize。

* DISPLAY_FREQ controls the frequency of printing information in running.

* DATASET_TIMER_ENABLED and DATASET_TIMER_THRESHOLD controls showing the time about dataset related operatioins.

  .. code-block:: python

     os.environ['BATCH_SIZE'] = str(batch_size)
     os.environ['DISPLAY_FREQ'] = str(self.config['args'].get('display', 1)) # The interval of iterations for showing.
     if cfg_dataset_timer: # The time evalutor for datasets.
         os.environ['DATASET_TIMER_ENABLED'] = str(1 if cfg_dataset_timer['enabled'] is True else 0)
         os.environ['DATASET_TIMER_THRESHOLD'] = str(cfg_dataset_timer['threshold_seconds'])

up.tasks.__init__
-----------------

UP adds DEFAULT_TASKS and EXCLUDE_TASKS into environment variables to control the loading of tasks.

* DEFAULT_TASKS: all tasks under the 'tasks' folder.

* EXCLUDE_TASKS: the task which will not be loaded.

  .. code-block:: python

     pwd = os.path.dirname(os.path.realpath(__file__))
     tasks_names = os.environ.get("DEFAULT_TASKS", os.listdir(pwd)) # loading all tasks.
     exclude_tasks = os.environ.get("EXCLUDE_TASKS", '').split(":") # excluding the writing task.

up.utils.general.petrel_helper
------------------------------

UP adds PETRELPATH into environment variables to control the loading of softwares.

  .. code-block:: python

     default_conf_path = os.environ.get('PETRELPATH', '~/petreloss.conf')

up.utils.general.registry
-------------------------

UP adds REGTRACE into environment variables to control the registers.

  .. code-block:: python

     _REG_TRACE_IS_ON = os.environ.get('REGTRACE', 'OFF').upper() == 'ON'
