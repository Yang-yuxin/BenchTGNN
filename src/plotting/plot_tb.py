import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_tensorboard_data(event_file_path, tag_name):
    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(event_file_path,
        size_guidance={event_accumulator.SCALARS: 0})

    # Load the data from the file
    ea.Reload()

    # Extract the scalar data for the specified tag
    scalar_data = ea.Scalars(tag_name)

    # Convert to numpy arrays for ease of use
    steps = np.array([s.step for s in scalar_data])
    values = np.array([s.value for s in scalar_data])

    return steps, values

def plot_data(steps, values, tag_name):
    plt.plot(steps, values, label=tag_name)
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title(f'{tag_name}')
    # plt.legend()

def list_tags(event_file_path):
    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(event_file_path,
        size_guidance={event_accumulator.SCALARS: 0})

    # Load the data from the file
    ea.Reload()

    # Print all scalar tags
    print("Available Scalar Tags:", ea.Tags()["scalars"])

plt.figure(figsize=(10, 6))
# Usage
event_file_path = []
event_file_path.append('log_tb/log_tbWIKI_memory_11-28 05:10:14/events.out.tfevents.1701177014.sutlej.1669684.0')
event_file_path.append('log_tb/log_tbWIKI_memory_11-28 06:23:28/events.out.tfevents.1701181408.sutlej.1675131.0')
event_file_path.append('log_tb/log_tbWIKI_memory_11-28 06:25:09/events.out.tfevents.1701181509.sutlej.1675443.0')
event_file_path.append('log_tb/log_tbWIKI_memory_11-28 09:05:26/events.out.tfevents.1701191126.sutlej.1685299.0')
event_file_path.append('log_tb/log_tbWIKI_memory_11-28 10:21:31/events.out.tfevents.1701195691.sutlej.1690143.0')
tag_name = 'MRR/Val'  # Replace with your tag name

# list_tags(event_file_path)
for path in event_file_path:
    steps, values = extract_tensorboard_data(path, tag_name)
    plot_data(steps, values, tag_name)
plt.savefig('tb_mrr.png')