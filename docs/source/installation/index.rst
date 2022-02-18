安装说明/Installation
=====================

1. 查询 tags 查看历史版本，推荐使用 master 分支

.. code-block:: bash

     git clone git@gitlab.bj.sensetime.com:spring2/universal-perception.git UP 
     cd UP

2. 激活环境：激活集群 Pytorch 环境

.. code-block:: bash

     source s0.3.4 or s0.3.3

3. 编译代码，在集群环境内可采用srun或spring.submit run完成，示例如下：

srun模式:

.. code-block:: bash

     srun -p $partition --gres=gpu:1 python setup.py build_ext -i

spring.submit run模式:

.. code-block:: bash

    spring.submit run -p $partition -n1 --gpu "python setup.py build_ext -i"

4. 使用脚本编译，我们提供了easy_setup.sh脚本方便直接编译代码：

.. code-block:: bash

    ./easy_setup.sh $partition

.. note::

    * <$partition> 为集群分区名，可通过sinfo命令查看
