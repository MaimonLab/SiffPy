Installing ``SiffPy``
=================================================================================================

``SiffPy`` can be installed with ``pip`` or ``conda``. The ``conda`` recipe
is NOT available on ``conda-forge`` (maybe I'll make it there when the
``ScanImage-FLIM`` package is public) but can be installed if you're
on the Rockefeller network from the ``maimon-forge`` directory.

----------
``pip``
----------

If you're going to install it in a specific environment, first
activate that environment so that your ``pip`` command is pointing
to the right place. Then navigate to the directory containing the
``SiffPy`` source and run

.. code-block:: console
    
    (.venv) $ pip install .

----------
``conda``
----------

It can be installed in a ``conda`` environment using the ``maimon-forge``
channel.

.. code-block:: console
    
    (.venv) $ conda install siffpy -c <path_to_Maimon_server>/lab_resources/maimon-forge

Or you can make its own environment and install it there:

.. code-block:: console
    
    conda create --name siffpy python=3.9
    conda activate siffpy
    git clone git@github.com:MaimonLab/SiffPy.git
    PYTHON -m pip install SiffPy/.