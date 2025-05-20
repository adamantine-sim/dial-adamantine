This is a test application for fitting a Dial surrogate to MDF data for the dimensions of an LPBF melt pool.

See the Dial repo for more information:
https://github.com/INTERSECT-DIAL/dial

## Quickstart skeleton
1) Install the INTERSECT-SDK

`pip install intersect-sdk`

2) Install the Dial data class

_Eventually just the data classes will be in their own package, but for now you have to fully install Dial locally._

`pip install git+https://github.com/INTERSECT-DIAL/dial`

3) Clone the repo

`git clone https://code.ornl.gov/intersect/dialed/dial-am-melt-pool.git`

4) Go into the root folder

`cd dial-am-melt-pool`

5) Run the client script

`python am-mp-client.py --config remote-conf.json`