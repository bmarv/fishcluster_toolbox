# pre-processing steps
1. Environment
    1. Motionmapper Installation as intermediate step until fully embedded:
    ```bash 
    pip install -r requirements_mmpy.txt
    cd ../motionmapperpy
    python setup.py install
    cd ../
    ```
    2. Python Packages
    ```bash
    pip install -r requirements.txt
    ```
    3. build cython module
    ```bash
    cython utils.processing_methods.pyx
    ```
2. setup environment vars:
    * config.env
        - projectPath
        - BLOCK1
        - BLOCK2
        - BLOCK
    * config_processing.env
        - path_csv_local
        - POSITION_STR_FRONT
        - POSITION_STR_BACK
3. run preprocessing each Block individually
    ```bash
    python processing.data_processing
    ```