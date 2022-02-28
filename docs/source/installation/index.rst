安装说明/Installation
=====================

1. 查询 tags 查看历史版本，推荐使用 master 分支

.. code-block:: bash

    git clone git@gitlab.bj.sensetime.com:spring2/united-perception.git UP 
    cd UP

2. 激活环境：激活集群 Pytorch 环境

.. code-block:: bash

    source s0.3.4 or s0.3.3

3. 编译代码，在集群环境内可采用srun或spring.submit run完成，示例如下：

srun模式:

.. code-block:: bash

    #!/bin/bash

    export PATH=/mnt/lustre/share/gcc/gcc-5.3.0/bin/:$PATH
    export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST='3.5;5.0+PTX;6.0;7.0'

    pip uninstall -y nart_tools
    pip uninstall -y nart==0.2.4
    pip uninstall -y torchvision
    pip uninstall -y springvision==1.0.1
    pip uninstall -y kestrel==1.5.4-patch1
    pip install --user -r requirements.txt

    partition=$1
    srun -p $partition --gres=gpu:1 python setup.py build_ext -i

spring.submit run模式:

.. code-block:: bash

    #!/bin/bash

    export PATH=/mnt/lustre/share/gcc/gcc-5.3.0/bin/:$PATH
    export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:/mnt/lustre/share/gcc/mpfr-2.4.2/lib:/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST='3.5;5.0+PTX;6.0;7.0'

    pip uninstall -y nart_tools
    pip uninstall -y nart==0.2.4
    pip uninstall -y torchvision
    pip uninstall -y springvision==1.0.1
    pip uninstall -y kestrel==1.5.4-patch1
    pip install --user -r requirements.txt

    partition=$1
    spring.submit run -p $partition -n1 --gpu "python setup.py build_ext -i"

.. note::

    * pip install时可能造成环境与其他框架冲突，此时建议使用pip install -t $PATH -r requirements.txt，PATH为任一目录用于存放UP所需环境，在使用时，将$PATH添加至PYTHONPATH即可

4. 使用脚本编译，我们提供了easy_setup.sh与easy_setup_spring.sh脚本方便直接编译代码：

.. code-block:: bash

    # srun 编译脚本
    ./easy_setup.sh $partition
    # spring.submit 编译脚本
    ./easy_setup_spring.sh $partition

.. note::

    * 建议使用已有脚本编译
    * <$partition> 为集群分区名，可通过sinfo命令查看
