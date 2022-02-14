安装说明/Installation
=====================

1. 查询 tags 查看历史版本，推荐使用 master 分支

.. code-block:: bash

     git clone git@gitlab.bj.sensetime.com:spring2/universal-perception.git UP 
     cd UP

2. 激活环境：激活集群 Pytorch 环境

.. code-block:: bash

     source s0.3.4 or s0.3.3

通过脚本完成安装

.. code-block:: bash

     ./easy_setup.sh <partition>

<partition> 为集群可通过 sinfo 命令查看
