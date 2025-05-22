This is an INTERSECT workflow to optimize Adamantine simulations using Dial.

See the Dial repo for more information:
https://github.com/INTERSECT-DIAL/dial

## Quickstart skeleton (Remote INTERSECT Core)
1) Install the INTERSECT-SDK

`pip install intersect-sdk`

2) Install the Dial data class

_Eventually just the data classes will be in their own package, but for now you have to fully install Dial locally._

`pip install git+https://github.com/INTERSECT-DIAL/dial`

3) Clone the repo

`git clone https://github.com/adamantine-sim/dial-adamantine.git`

4) Go into the root folder

`cd dial-adamantine/jun25_dwell_rook`

5) Create utility and scratch directories

`mkdir utils`
`mkdir scratch`

6) Clone the adamantine workflow scripts repo in the utils directory

`git clone https://github.com/adamantine-sim/workflow-scripts.git workflow_scripts`

7) Run the client script

`python jun25-adamantine-client.py --config remote-conf.json`

## Quickstart skeleton (Remote INTERSECT Core)
1) Install the INTERSECT-SDK

`pip install intersect-sdk`

2) Clone and deploy Dial and INTERSECT Core Services locally

`git clone https://github.com/INTERSECT-DIAL/dial`
`cd dial`
`docker compose up -d`
`cd scripts`
`python launch_service.py`

3) Clone the repo

`git clone https://github.com/adamantine-sim/dial-adamantine.git`

4) Go into the root folder

`cd dial-adamantine/jun25_dwell_rook`

5) Create utility and scratch directories

`mkdir utils`
`mkdir scratch`

6) Clone the adamantine workflow scripts repo in the utils directory

`git clone https://github.com/adamantine-sim/workflow-scripts.git workflow_scripts`

7) Run the client script

`python jun25-adamantine-client.py --config local-conf.json`
