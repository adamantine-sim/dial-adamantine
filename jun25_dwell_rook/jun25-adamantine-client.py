import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

import docker
from scipy.stats import qmc
import shutil
import time

from utils.workflow_scripts.jun25_dwell_rook_toolpath.toolpath_writer import write_toolpath
from utils.workflow_scripts.analysis_17_4_PH.analysis_17_4_PH import analysis


mpl.use('agg')

import numpy as np
from intersect_sdk import (
    INTERSECT_JSON_VALUE,
    HierarchyConfig,
    IntersectClient,
    IntersectClientCallback,
    IntersectClientConfig,
    IntersectDirectMessageParams,
    default_intersect_lifecycle_loop,
)

# from scipy.stats import qmc
from dial_dataclass import (
    DialInputPredictions,
    DialInputSingleOtherStrategy,
    DialWorkflowCreationParamsClient,
    DialWorkflowDatasetUpdate,
)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

# MANUAL INPUTS ------------------------------------------------------------------------------------------------------

#INITIAL_BOUNDS = [[10,60], [10,60]]
# four dwell‐bottom chunks, four reheat‐temp chunks, one final dwell
dwell0_bounds      = [[10,60] for _ in range(4)]  # dwell_0 chunked into 4 zones
reheat_temp_bounds = [[0,900] for _ in range(4)]  # reheat temp chunked into 4 zones
dwell1_bounds      = [[0,60]]  # final dwell one zone
INITIAL_BOUNDS     = dwell0_bounds + reheat_temp_bounds + dwell1_bounds
NUM_DIMS = len(INITIAL_BOUNDS)
MESHGRID_SIZE = 101
#INITIAL_MESHGRIDS = np.meshgrid(
#    *[np.linspace(dim_bounds[0], dim_bounds[1], MESHGRID_SIZE) for dim_bounds in INITIAL_BOUNDS],
#    indexing='ij',
#)
#INITIAL_POINTS_TO_PREDICT = np.hstack([mg.reshape(-1, 1) for mg in INITIAL_MESHGRIDS])
SLICE_DIMS = (0, 4)
MESHGRID_SIZE = 101

# Build a 2D mesh just for those two dims:
low_high = [INITIAL_BOUNDS[i] for i in SLICE_DIMS]
mgx, mgy = np.meshgrid(
    np.linspace(low_high[0][0], low_high[0][1], MESHGRID_SIZE),
    np.linspace(low_high[1][0], low_high[1][1], MESHGRID_SIZE),
    indexing='ij'
)

# Now we need to pack those into full 9-D points by fixing the OTHER dims at midpoints:
fixed_vals = {
    i: 0.5*(b[0]+b[1])
    for i, b in enumerate(INITIAL_BOUNDS)
    if i not in SLICE_DIMS
}
pts = []
for xi, yi in zip(mgx.ravel(), mgy.ravel()):
    full = np.zeros(NUM_DIMS)
    full[SLICE_DIMS[0]] = xi
    full[SLICE_DIMS[1]] = yi
    for j, v in fixed_vals.items():
        full[j] = v
    pts.append(full)
INITIAL_POINTS_TO_PREDICT = np.vstack(pts)

NUM_ITERATIONS = 200
INITIAL_DATA_SIZE = 90


# ADAMANTINE SPECIFIC FUNCTIONS ------------------------------------------------------------------------------------------------------
directory = os.path.dirname(os.path.realpath(__file__))

'''
Here, we attempt to find the optimal dwell times to create the dwell rook. In the initial version, we're just setting the length of one 
of the dwells per layer split with different values for the first half and second half of the print.

Note that the service must be started first, then the client.
'''

# Global path variables
scratch_path = os.path.join(directory,'scratch')
adamantine_filename = "solution_dwell"
input_filename = "input.info"
print_path = "print_layers"
reheat_path = "reheat_layers"
field_name = 'temperature'
line_plots = False
volume_plot = False

# Functions to run and analyze the simulation
def run_adamantine(mount_path_host):
    image_name = "rombur/adamantine:latest"
    client = docker.from_env()
    mount_path_container = "/home/volume"
    mount_string = [mount_path_host+":"+mount_path_container]
    run_command = "bash /home/volume/commands.sh 1"
    container_logs = client.containers.run(image_name, run_command, volumes=mount_string, detach=False, stderr=True)
    return

def run_detached_adamantine(mount_path_host,idx):
    image_name = "rombur/adamantine:latest"
    client = docker.from_env()
    mount_path_container = "/home/volume"
    mount_string = [mount_path_host+":"+mount_path_container]
    run_command = "bash /home/volume/commands.sh 1"
    n_cpus = os.cpu_count()
    cpu_to_use = idx % n_cpus
    container = client.containers.run(image_name, run_command, volumes=mount_string, detach=True, stderr=True)
    return container

def analyze_results(volume_path):
    score = None
    # Call functions from workflow-scripts
    score = analysis(volume_path, adamantine_filename, field_name, line_plots, volume_plot)
    return score


# Functions to set up the environment
def setup():
    print("Starting setup...")

    # Create the scratch volume if it doesn't already exist
    scratch_volume_name = "scratch_volume"
    volume_path = os.path.join(scratch_path, scratch_volume_name)
    if not os.path.exists(volume_path):
        os.makedirs(volume_path)

    run_script_filename = "commands.sh"

    shutil.copyfile(input_filename, os.path.join(volume_path, input_filename))
    shutil.copyfile(run_script_filename, os.path.join(volume_path, run_script_filename))
    shutil.copyfile("mesh.inp", os.path.join(volume_path, "mesh.inp"))

    print("Complete.")

    return volume_path

def get_toolpath(volume_path, parameters):
    toolpath_info = {}
    toolpath_info['print_path'] = print_path
    toolpath_info['reheat_path'] = reheat_path
    toolpath_info['reheat_power'] = parameters['reheat_power'] # W
    toolpath_info['scan_path_out'] = "scan_path.inp"
    toolpath_info['lump_size'] = 2
    toolpath_info['dwell_0'] = parameters['dwell_0']
    toolpath_info['dwell_1'] = parameters['dwell_1'] 
    write_toolpath(toolpath_info)
    toolpath_filename = "scan_path.inp"
    shutil.copyfile(toolpath_filename, os.path.join(volume_path, toolpath_filename))


def get_data_point(x, mount_path_host):
    
    print("Getting data points...") 
    
    #toolpath_parameters = {}
    #toolpath_parameters['dwell_bottom'] = x[0]
    #toolpath_parameters['dwell_top'] = x[1]
    
    toolpath_parameters = {
        'dwell_0'      : list(x[0:4]),
        'reheat_power' : list(x[4:8]),
        'dwell_1'      : [ x[8] ],
    }

    get_toolpath(mount_path_host, toolpath_parameters)

    run_adamantine(mount_path_host)
    score = analyze_results(mount_path_host)

    # Dummy score for testing
    #score = x[0]*x[0] + x[1]
    #time.sleep(30)
    
    print("Complete.")
    
    return score

def get_data_point_batch(x_batch, mount_path_host):

    print("Getting a batch of data points...")

    scores = []
    containers = []
    for batch_index in range(0,len(x_batch)):
        batch_dir = Path(mount_path_host + '/' + str(int(batch_index)))
        batch_dir.mkdir(parents=True, exist_ok=True)

        x = x_batch[batch_index]
        #toolpath_parameters = {}
        #toolpath_parameters['dwell_bottom'] = x[0]
        #toolpath_parameters['dwell_top'] = x[1]
        toolpath_parameters = {
            'dwell_0'      : list(x[0:4]),
            'reheat_power' : list(x[4:8]),
            'dwell_1'      : [ x[8] ],
        }


        batch_dir = mount_path_host + '/' + str(int(batch_index))

        get_toolpath(batch_dir, toolpath_parameters)

        shutil.copyfile(os.path.join(mount_path_host, 'commands.sh'), os.path.join(batch_dir, 'commands.sh'))
        shutil.copyfile(os.path.join(mount_path_host, 'mesh.inp'), os.path.join(batch_dir, 'mesh.inp'))
        shutil.copyfile(os.path.join(mount_path_host, 'input.info'), os.path.join(batch_dir, 'input.info'))

        container = run_detached_adamantine(batch_dir, batch_index)
        containers.append(container)

    for c in containers:
        c.wait()

    for batch_index in range(0,len(x_batch)):
        batch_dir = Path(mount_path_host + '/' + str(int(batch_index)) + '/')
        score = analyze_results(batch_dir)

        # Dummy score for testing
        #x = x_batch[batch_index]
        #score = x[0]*x[0] + x[1]

        scores = scores + [score]

    print("Complete.")

    return scores

def graph(mean_grid, variance, dataset_x, dataset_y):
    plt.clf()
    plt.contourf(mgx, mgy, self.mean_grid, extend='both')
    cbar = plt.colorbar()
    cbar.set_label('Score')
    plt.xlabel('Dwell bottom (s)')
    plt.ylabel('Dwell top (s)')
    # overlay training points in the slice dims
    X_train = np.array(dataset_x)
    x_vals  = X_train[:, SLICE_DIMS[0]]
    y_vals  = X_train[:, SLICE_DIMS[1]]
    plt.scatter(
        x_vals, y_vals,
        facecolors='none', edgecolors='black', marker='o',
        label='Training points'
    )
    # last BO suggestion in red
    x_rec, y_rec = dataset_x[-1][SLICE_DIMS[0]], dataset_x[-1][SLICE_DIMS[1]]
    plt.scatter(
        x_rec, y_rec,
        color='red', marker='x', s=100,
        label='BO next point'
    )
    plt.legend(loc='best')
    plt.savefig('function_value.png')
    
    # ---- Predictive VARIANCE contour ----
    plt.clf()
    cf = plt.contourf(mgx, mgy, variance, extend='both')
    cbar = plt.colorbar(cf)
    cbar.set_label('Predictive variance')
    plt.xlabel(f'Dim {SLICE_DIMS[0]}')
    plt.ylabel(f'Dim {SLICE_DIMS[1]}')

    # reuse the same overlay
    plt.scatter(x_vals, y_vals, facecolors='none', edgecolors='black', marker='o')
    plt.scatter(x_next, y_next, color='red', marker='x', s=100)

    plt.savefig('function_variance_slice.png')
    
    #---- 1D LINE-OUT through variance/mean at fixed y-index ----
    # Choose the slice index along the second axis (e.g. first column)
    j = 0
    xs = mgx[:, j]
    mu = mean_grid[:, j]
    sigma = np.sqrt(variance[:, j])

    plt.clf()
    plt.plot(xs, mu, '-', label='Mean')
    plt.fill_between(xs, mu - sigma, mu + sigma, alpha=0.3, label='±1σ')

    # Overlay training points that lie at that fixed y-value (within tol)
    tol = 1e-6
    for xi, yi, yi_score in zip(x_vals, y_vals, dataset_y):
        if abs(yi - mgy[0, j]) < tol:
            plt.plot(xi, yi_score, 'ko')  # black circle

    # Overlay the next suggestion if it’s on that slice
    if abs(y_next - mgy[0, j]) < tol:
        # assume you have its score in dataset_y[-1]
        plt.plot(x_next, dataset_y[-1], 'rx', markersize=10)

    plt.xlabel(f'Dim {SLICE_DIMS[0]}')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.savefig('lineout_slice.png')
    
    for dim in range(4):
        # fix all other dims (including your reheat chunks and dwell_1) at midpoints
        xs, mu, sigma = get_lineout_for_dim(dim)
        plt.figure()
        plt.plot(xs, mu, '-')
        plt.fill_between(xs, mu-sigma, mu+sigma, alpha=0.3)
        plt.title(f'Line-out along dim {dim}')
        plt.xlabel(f'Dim {dim}')
        plt.ylabel('Score')
        plt.savefig(f'lineout_dim{dim}.png')
    
# ORCHESTRATOR ------------------------------------------------------------------------------------------------------

class ActiveLearningOrchestrator:
    def __init__(self, service_destination: str):
        self.service_destination = service_destination

        self.num_dims = NUM_DIMS
        self.bounds = INITIAL_BOUNDS
        self.initial_data_size = INITIAL_DATA_SIZE

        # This value gets populated from the return value of initializing the workflow
        self.workflow_id = ''

        self.volume_path = setup()

        # We use the following Latin Hypercube Sampling to generate self.dataset_x:
        self.rng = np.random.default_rng(seed=42)
        self.lhs_sampler = qmc.LatinHypercube(d=self.num_dims, seed=self.rng)
        self.unscaled_lhs = self.lhs_sampler.random(n=self.initial_data_size)
        self.l_bounds = [bound[0] for bound in self.bounds]
        self.u_bounds = [bound[1] for bound in self.bounds]
        self.dataset_x = qmc.scale(self.unscaled_lhs, self.l_bounds, self.u_bounds).tolist()
        
        self.dataset_y = []
    
        #for x_val in self.dataset_x:
        #     print("XVAL:", x_val)
        #     y_val = get_data_point(x_val, self.volume_path)
        #     print("YVAL:", y_val)
        #     self.dataset_y.extend(y_val)

        # NOTE: Currently this assigns exactly one CPU core to each process. This could be an issue for large simulations 
        # (want more than one core) or for a large batch (if the batch size is larger than the number of cores).
        self.dataset_y = get_data_point_batch(self.dataset_x, self.volume_path)

        print("DATASET X: ",self.dataset_x)
        print("DATASET Y: ",self.dataset_y)


    # create a message to send to the server
    def assemble_message(self, operation: str, **kwargs: Any) -> IntersectClientCallback:
        if operation == 'initialize_workflow':
            print("operation is initialize_workflow")

            payload = DialWorkflowCreationParamsClient(
                dataset_x=self.dataset_x,
                dataset_y=self.dataset_y,
                bounds=INITIAL_BOUNDS,
                kernel='rbf',
                length_per_dimension=True,  # allow the matern to use separate length scales for the two parameters
                y_is_good=True,  
                backend='sklearn',  # "sklearn" or "gpax"
                seed=-1,  # Use seed = -1 for random results
                preprocess_standardize=False
            )
        elif operation == 'update_workflow_with_data':
            print("operation is update_workflow_with_data")
            payload = DialWorkflowDatasetUpdate(
                workflow_id=self.workflow_id,
                **kwargs,
            )
        elif operation == 'get_next_point':
            print("operation is get_next_point")
            payload = DialInputSingleOtherStrategy(
                workflow_id=self.workflow_id,
                strategy='expected_improvement',
            )
        elif operation == 'get_surrogate_values':
            print("operation is get_surrogate_values")
            payload = DialInputPredictions(
                workflow_id=self.workflow_id,
                points_to_predict=INITIAL_POINTS_TO_PREDICT,
            )
        else:
            err_msg = f'Invalid operation {operation}'
            raise Exception(err_msg)  # noqa: TRY002
        
        print("Sending message...")

        return IntersectClientCallback(
            messages_to_send=[
                IntersectDirectMessageParams(
                    destination=self.service_destination,
                    operation=f'dial.{operation}',
                    payload=payload,
                )
            ]
        )

    # The callback function.  This is called whenever the server responds to our message.
    # This could instead be implemented by defining a callback method (and passing it later), but here we chose to directly make the object callable.
    def __call__(
        self,
        _source: str,
        operation: str,
        has_error: bool,
        payload: INTERSECT_JSON_VALUE,
    ) -> IntersectClientCallback:
        if has_error:
            print('============ERROR==============', file=sys.stderr)
            print(operation, file=sys.stderr)
            print(payload, file=sys.stderr)
            print(file=sys.stderr)
            raise Exception  # noqa: TRY002 (break INTERSECT loop)
        if operation == 'dial.initialize_workflow':
            self.workflow_id: str = payload
            return self.assemble_message('get_surrogate_values')
        if operation == 'dial.update_workflow_with_data':
            return self.assemble_message('get_surrogate_values')
        if (operation == 'dial.get_surrogate_values'):  # if we receive a grid of surrogate values, record it for graphing, then ask for the next recommended point
            self.mean_grid = np.array(payload[0]).reshape((MESHGRID_SIZE,MESHGRID_SIZE))
            self.variance = np.array(payload[1]).reshape((MESHGRID_SIZE,MESHGRID_SIZE))
            return self.assemble_message('get_next_point')
            
        if operation == 'dial.get_next_point':
            self.dataset_x.append(payload)
            coord_str = ', '.join([f'{coord:.2f}' for coord in payload])
            print(f'Running simulation at ({coord_str}): ', end='', flush=True)
            
            
            y = get_data_point(payload, self.volume_path)
            #y = get_data_point_batch([payload], self.volume_path)[0]
            self.dataset_y.append(y)

            graph(self.mean_grid, self.variance, self.dataset_x, self.dataset_y)

            if len(self.dataset_y) > NUM_ITERATIONS:
                raise Exception # noqa: TRY002 (INTERSECT interaction mechanism, do not need custom exception)
            
            print("NEXT X", payload)
            print("NEXT Y", y)
            
            return self.assemble_message('update_workflow_with_data', next_x=payload, next_y=float(y))

        err_msg = f'Unknown operation received: {operation}'
        raise Exception(err_msg)  # noqa: TRY002 (INTERSECT interaction mechanism)

# MAIN ------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # In production, everything in this dictionary should come from a configuration file, command line arguments, or environment variables.
    parser = argparse.ArgumentParser(description='jun25 dwell rook client')
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    try:
        with Path(args.config).open('rb') as f:
            from_config_file = json.load(f)
    except (json.decoder.JSONDecodeError, OSError) as e:
        logger.critical('unable to load config file: %s', str(e))
        sys.exit(1)
    active_learning = ActiveLearningOrchestrator(
        service_destination=HierarchyConfig(
            **from_config_file['intersect-hierarchy']
        ).hierarchy_string('.')
    )
    config = IntersectClientConfig(
        initial_message_event_config=active_learning.assemble_message('initialize_workflow'),
        **from_config_file['intersect'],
    )

    # use the orchestator to create the client
    client = IntersectClient(
        config=config,
        user_callback=active_learning,  # the callback (here we use a callable object, as discussed above)
    )

    # This will run the send message -> wait for response -> callback -> repeat cycle until we have 25 points (and then raise an Exception)
    default_intersect_lifecycle_loop(
        client,
    )
