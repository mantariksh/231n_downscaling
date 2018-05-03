climateAI-forecast
##################

Software to develop deep learning models to make seasonal forecasts of climate phenomena.

For example a model could be developed to predict an ENSO index - `Nino3.4 <https://iridl.ldeo.columbia.edu/maproom/ENSO/Diagnostics.html>`_ - at a 1 year lead time. To train the model we use thousands of years of simulated data from climate models (GCMs) to establish relationships between spatiotemporal patterns of global climate variables and the state of Nino3.4. 

Installation
------------

Python 3.6

.. code-block:: bash

    $ pip3 install -r requirements.txt
    
Data
----

GCM data is available from `ESGF <https://esgf-data.dkrz.de/search/cmip5-dkrz/>`_ in netcdf file format. Opening and working with a set of NetCDF files is handled by the python package `xarray <http://xarray.pydata.org/en/stable/>`_. Xarray uses dask to lazily execute computation to avoid overloading memory. NetCDF files can be conveniently viewed using `Panoply <https://www.giss.nasa.gov/tools/panoply/>`_. 

The data best suited for seasonal climate forecasting is monthly resolution. Some of the variables that are generally thought to be important include surface temperature (tas), precipitation (pr), sea level pressure (psl), geopotential height (zg), and u and v components of surface wind velocity (uas, vas).

Data Preprocessing
==================

The script ``prepreprocess_gcm.py`` performs three data preprocessing tasks. 

1. regridding GCM to a common spatial grid and computing anomalies
2. computing a response variable time series by taking the mean of a spatial subset for a specific variable
3. writing individual image files for each time step

The script requires a specific file structure to work. The root directory, ``data_dir``, must be specified in the script. ``data_dir`` has a subdirectory `GCMs` which contains gcm data organized by `gcm_name` and `scenario` like ``GCMs/gcm_name/scenario``. For each scenario there is a `raw` directory with the original netcdf files, a `regrid_anomalies` directory with the climate variable anomalies interpolated onto a common grid, and an `img` directory with the individual timesteps saved as image or numpy files.

.. code-block:: bash

    | `data_dir`
    |-- GCMs
    |---- `gcm_name`
    |------ `scenario`
    |-------- raw
    |-------- regrid_anomalies
    |-------- img


The script has several settings that can be changed. See the docstrings and comments in the script for details.

