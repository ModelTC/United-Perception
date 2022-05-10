Installation
============

1. Query tags to view historical versions. 'Master' branch is recommended.

.. code-block:: bash

    git clone git@gitlab.bj.sensetime.com:spring2/united-perception.git UP 
    cd UP

2. Activate the pytorch environment.

.. code-block:: bash

    source s0.3.4 or s0.3.3

3. Compiling the code, which can be finished by srun or spring.submit run in the server as followed.

srun:

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
    srun -p $partition -n1 --gres=gpu:1 python setup.py build_ext -i

spring.submit run:

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

    * pip install may make the environment conflict with other frameworks, and thus we recommand using pip install -t $PATH -r requirements.txt where PATH is any dir for saving the required environment of UP. Please adding $PATH into PYTHONPATH.

4. We support easy_setup.sh and easy_setup_spring.sh for directly compiling by scripts：

.. code-block:: bash

    # compiling by srun.
    ./easy_setup.sh $partition
    # compiling by spring.submit.
    ./easy_setup_spring.sh $partition

.. note::

    * recommend using the given scripts. 
    * <$partition> is the name of partition of the sever which can be viewed by sinfo command.
