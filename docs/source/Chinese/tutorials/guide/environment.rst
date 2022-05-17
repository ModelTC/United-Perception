环境变量
========

介绍通过改变环境变量可以实现的功能。环境变量通过如下方式改变。

  .. code-block:: bash

     export xxx=xxxx

up.__init__
-----------

UP 通过将 plugins 加入环境变量来读取路径。

  .. code-block:: python

     _PPPATH = 'PLUGINPATH'
     if _PPPATH not in os.environ:
         return
     path_list = os.environ[_PPPATH].split(':')

up.runner.base_runner
---------------------

UP 将 batchsize, display, dataset_timer 加入环境变量。

* BATCH_SIZE 用于控制 batchsize。

* DISPLAY_FREQ 用于控制运行中的信息显示频率。

* DATASET_TIMER_ENABLED, DATASET_TIMER_THRESHOLD 用来控制数据集相关时间的显示。

  .. code-block:: python

     os.environ['BATCH_SIZE'] = str(batch_size)
     os.environ['DISPLAY_FREQ'] = str(self.config['args'].get('display', 1)) # The interval of iterations for showing.
     if cfg_dataset_timer: # The time evalutor for datasets.
         os.environ['DATASET_TIMER_ENABLED'] = str(1 if cfg_dataset_timer['enabled'] is True else 0)
         os.environ['DATASET_TIMER_THRESHOLD'] = str(cfg_dataset_timer['threshold_seconds'])

up.tasks.__init__
-----------------

UP 通过将 DEFAULT_TASKS, EXCLUDE_TASKS 加入环境变量来控制任务加载。

* DEFAULT_TASKS 在 tasks 目录下的所有任务。

* EXCLUDE_TASKS 不被读取的任务类型。

  .. code-block:: python

     pwd = os.path.dirname(os.path.realpath(__file__))
     tasks_names = os.environ.get("DEFAULT_TASKS", os.listdir(pwd)) # loading all tasks.
     exclude_tasks = os.environ.get("EXCLUDE_TASKS", '').split(":") # excluding the writing task.

up.utils.general.petrel_helper
------------------------------

UP 将 PETRELPATH 加入环境变量来控制软件。

  .. code-block:: python

     default_conf_path = os.environ.get('PETRELPATH', '~/petreloss.conf')

up.utils.general.registry
-------------------------

UP 将 REGTRACE 加入环境变量来控制注册器。

  .. code-block:: python

     _REG_TRACE_IS_ON = os.environ.get('REGTRACE', 'OFF').upper() == 'ON'
