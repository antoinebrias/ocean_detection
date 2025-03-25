import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import ipywidgets as widgets
from IPython.display import display
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class Cetacean:
    def __init__(self, xlim, ylim, step_size=1.0, marked=False):
        self.x = np.random.uniform(0, xlim)
        self.y = np.random.uniform(0, ylim)
        self.xlim = xlim
        self.ylim = ylim
        self.step_size = step_size
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.marked = marked  # Whether the cetacean is GPS-tracked
        self.color = 'red' if marked else 'grey'  # Red for marked cetaceans, grey for unmarked
        self.trajectory = [(self.x, self.y)]  # Track the cetacean's trajectory over time

    def move(self, neighbors, correlation_strength=0.4):
        random_angle = self.angle + np.random.uniform(-np.pi/4, np.pi/4)

        if neighbors:
            avg_angle = np.mean([n.angle for n in neighbors])
            self.angle = (1 - correlation_strength) * random_angle + correlation_strength * avg_angle
        else:
            self.angle = random_angle

        dx = self.step_size * np.cos(self.angle)
        dy = self.step_size * np.sin(self.angle)

        self.x = np.clip(self.x + dx, 0, self.xlim)
        self.y = np.clip(self.y + dy, 0, self.ylim)

        # Append the new position to the trajectory
        self.trajectory.append((self.x, self.y))

    def get_position(self):
        return self.x, self.y

# Hydrophone class
class Hydrophone:
    def __init__(self, x, y, detection_range=50.0):
        self.x = x
        self.y = y
        self.detection_range = detection_range
        self.detected_cetaceans = []  # List to store detected cetaceans
        self.detected = False  # New attribute to track detection status

    def detect(self, cetaceans):
        self.detected_cetaceans = []  # Reset detected cetaceans each step
        self.detected = False  # Reset detection status each step
        
        for cetacean in cetaceans:
            distance = np.linalg.norm([cetacean.x - self.x, cetacean.y - self.y])
            if distance <= self.detection_range:
                self.detected_cetaceans.append(cetacean)
                self.detected = True  # Set to True if anything is detected

# Simulation parameters
N = 50   # Total cetaceans
M = 5    # Marked cetaceans with GPS tracking
correlation_strength = 0.4  # Degree of movement correlation
xlim, ylim = 400, 400  # Ocean space dimensions
steps = 100  # Default number of time steps

# Initialize cetaceans with marked ones being red and unmarked grey
cetaceans = [Cetacean(xlim, ylim, marked=(i < M)) for i in range(N)]


def is_far_enough(new_pos, existing_positions, min_dist):
    """Check if new_pos is at least min_dist away from all existing_positions."""
    return all(np.linalg.norm(np.array(new_pos) - np.array(pos)) >= min_dist for pos in existing_positions)


# Initialize list to store density and errors at each step
density_over_time = []
errors_over_time = []

# Run simulation
def run_simulation(N=50, M=5, correlation_strength=0.4, xlim=400, ylim=400, steps=10, num_hydrophones=5, detection_range = 50):
    # Initialize cetaceans with marked ones being red and unmarked grey
    cetaceans = [Cetacean(xlim, ylim, marked=(i < M)) for i in range(N)]

    # Place hydrophones ensuring minimum separation
    min_distance = min(xlim, ylim) / num_hydrophones  # Adjust as needed
    hydrophones = []
    attempts = 0
    max_attempts = 100 * num_hydrophones  # Avoid infinite loops

    while len(hydrophones) < num_hydrophones and attempts < max_attempts:
        new_x = np.random.uniform(0, xlim)
        new_y = np.random.uniform(0, ylim)
        if is_far_enough((new_x, new_y), hydrophones, min_distance):
            hydrophones.append((new_x, new_y))
        attempts += 1

    # Convert to Hydrophone objects
    hydrophones = [Hydrophone(x, y, detection_range) for x, y in hydrophones]

    # Initialize list to store density and errors at each step
    density_over_time = []
    errors_over_time = []

    # Run simulation
    for _ in range(steps):
        all_positions = []
        marked_positions = []
        detected_positions = []
        for i, cetacean in enumerate(cetaceans):
            neighbors = [c for c in cetaceans if np.linalg.norm([c.x - cetacean.x, c.y - cetacean.y]) < 20 and c != cetacean]
            cetacean.move(neighbors, correlation_strength)
            all_positions.append(cetacean.get_position())
            if cetacean.marked:
                marked_positions.append(cetacean.get_position())
        
        # Detect cetaceans for each hydrophone
        for hydrophone in hydrophones:
            hydrophone.detect(cetaceans)

        # Collect hydrophone-detected cetacean positions
        detected_positions = []
        for hydrophone in hydrophones:
            detected_positions.extend([(c.x, c.y) for c in hydrophone.detected_cetaceans])
        
        # Compute and store density at this step
        grid_x, grid_y = np.mgrid[0:xlim:100j, 0:ylim:100j]  # Grid for interpolation
        kde_all = gaussian_kde(np.array(all_positions).T)
        density_all = kde_all(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

        kde_marked = gaussian_kde(np.array(marked_positions).T)
        density_marked = kde_marked(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_y.shape)

        # Compute density for detected cetaceans only if there are detected positions
        if detected_positions:
            kde_detected = gaussian_kde(np.array(detected_positions).T)
            density_detected = kde_detected(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_y.shape)
        else:
            # If no detected positions, set the density to zero (or some empty array)
            density_detected = np.zeros_like(grid_y)  # Same shape as grid_y, filled with zeros

        # Combine marked cetaceans and detected positions for hydrophone + marked
        combined_positions = marked_positions + detected_positions
        kde_combined = gaussian_kde(np.array(combined_positions).T) if combined_positions else gaussian_kde(np.zeros((2, 1)))
        density_combined = kde_combined(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_y.shape)

  

        # Compute Mean Squared Error (MSE) between estimated and true density
        mse_marked = np.mean((density_marked - density_all) ** 2)
        mse_detected = np.mean((density_detected - density_all) ** 2)
        mse_combined = np.mean((density_combined - density_all) ** 2)

        # Store errors over time
        errors_over_time.append((mse_marked, mse_detected, mse_combined))

              # Compute and store density at this step and error
        density_over_time.append({
            'density_all': density_all.tolist(),
            'density_marked': density_marked.tolist(),
            'density_detected': density_detected.tolist(),
            'density_combined': density_combined.tolist(),
            'all_positions': [(x, y) for x, y in all_positions],
            'marked_positions': [(x, y) for x, y in marked_positions],
            'detected_positions': [(x, y) for x, y in detected_positions],
        })


    return density_over_time, cetaceans, hydrophones, errors_over_time



def plot_density_and_trajectory_at_time_step(step, density_over_time, errors_over_time, cetaceans, hydrophones, xlim, ylim):
    # Unpack the densities and error information
    step_data = density_over_time[step]  # Extract dictionary

    density_all = np.array(step_data['density_all'])  # Convert back to array
    density_marked = np.array(step_data['density_marked'])
    density_detected = np.array(step_data['density_detected'])
    density_combined = np.array(step_data['density_combined'])

    all_positions = step_data['all_positions']
    marked_positions = step_data['marked_positions']
    detected_positions = step_data['detected_positions']

    x_values = np.linspace(0, xlim, density_all.shape[1])  # Columns define x
    y_values = np.linspace(0, ylim, density_all.shape[0])  # Rows define y

  
    # Create the figure with 2 rows and 2 columns for the distributions, and 1 column for the errors
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=["True Distribution", "Marked Distribution", "Detected Distribution", "Combined Distribution", "MSE"],
        column_widths=[0.5, 0.5],  # Adjust width ratio for the error plot later
        row_heights=[0.33, 0.33, 0.33],  # Split rows evenly for the distribution plots
        shared_yaxes=False,
    )
    

    fig.add_trace(go.Heatmap(z=density_all.T, x = x_values, y = y_values, colorscale='Greens', showscale=False, name="True Distribution"), row=1, col=1)
    fig.add_trace(go.Heatmap(z=density_marked.T, x = x_values, y = y_values, colorscale='Reds', showscale=False, name="Marked Distribution"), row=1, col=2)
    fig.add_trace(go.Heatmap(z=density_detected.T, x = x_values, y = y_values, colorscale='Blues', showscale=False, name="Detected Distribution"), row=2, col=1)
    fig.add_trace(go.Heatmap(z=density_combined.T, x = x_values, y = y_values, colorscale='Purples', showscale=False, name="Combined Distribution"), row=2, col=2)

    # Plot the trajectories of only marked cetaceans on the density subplots
    for cetacean in cetaceans:
        if cetacean.marked:  # Only plot if the cetacean is marked
            trajectory_x, trajectory_y = zip(*cetacean.trajectory)
            fig.add_trace(go.Scatter(x=trajectory_x, y=trajectory_y, mode='lines', line=dict(color=cetacean.color, width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=trajectory_x, y=trajectory_y, mode='lines', line=dict(color=cetacean.color, width=1)), row=1, col=2)
            fig.add_trace(go.Scatter(x=trajectory_x, y=trajectory_y, mode='lines', line=dict(color=cetacean.color, width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=trajectory_x, y=trajectory_y, mode='lines', line=dict(color=cetacean.color, width=1)), row=2, col=2)

    # Plot cetacean positions (unmarked in grey with transparency, marked in red) on each distribution subplot
    all_positions_x, all_positions_y = zip(*all_positions)
    marked_positions_x, marked_positions_y = zip(*marked_positions)
    detected_positions_x, detected_positions_y = zip(*detected_positions)

    fig.add_trace(go.Scatter(x=all_positions_x, y=all_positions_y, mode='markers', marker=dict(color='grey', opacity=0.5, size=5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=marked_positions_x, y=marked_positions_y, mode='markers', marker=dict(color='red', size=5)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=all_positions_x, y=all_positions_y, mode='markers', marker=dict(color='grey', opacity=0.5, size=5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=marked_positions_x, y=marked_positions_y, mode='markers', marker=dict(color='red', size=5)), row=1, col=2)

    fig.add_trace(go.Scatter(x=all_positions_x, y=all_positions_y, mode='markers', marker=dict(color='grey', opacity=0.5, size=5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=marked_positions_x, y=marked_positions_y, mode='markers', marker=dict(color='red', size=5)), row=2, col=1)

    fig.add_trace(go.Scatter(x=all_positions_x, y=all_positions_y, mode='markers', marker=dict(color='grey', opacity=0.5, size=5)), row=2, col=2)
    fig.add_trace(go.Scatter(x=marked_positions_x, y=marked_positions_y, mode='markers', marker=dict(color='red', size=5)), row=2, col=2)



    # Plot hydrophone detections on each distribution subplot
    for hydrophone in hydrophones:
        hydrophone_x, hydrophone_y = hydrophone.x, hydrophone.y
        hydrophone_color = 'black'
        
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(color=hydrophone_color, size=10, symbol='x')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(color=hydrophone_color, size=10, symbol='x')), row=1, col=2)
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(color=hydrophone_color, size=10, symbol='x')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(color=hydrophone_color, size=10, symbol='x')), row=2, col=2)

        # Plot detection range as a circle around the hydrophone
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(size=hydrophone.detection_range, opacity=0.2, color='orange', line=dict(width=2)), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(size=hydrophone.detection_range, opacity=0.2, color='orange', line=dict(width=2)), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(size=hydrophone.detection_range, opacity=0.2, color='orange', line=dict(width=2)), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=[hydrophone_x], y=[hydrophone_y], mode='markers', marker=dict(size=hydrophone.detection_range, opacity=0.2, color='orange', line=dict(width=2)), showlegend=False), row=2, col=2)

        # Set axis limits for all subplots
    for i in range(1, 3):  # Row indices
        for j in range(1, 3):  # Column indices
            fig.update_xaxes(range=[0, xlim], row=i, col=j)
            fig.update_yaxes(range=[0, ylim], row=i, col=j)

    # Plot the errors over time using the provided plot_error_over_time function
    error_fig = plot_error_over_time(errors_over_time)
    
    # Add the error plot to the subplot (it will occupy the last column)
    fig.add_trace(error_fig.data[0], row=3, col=1)  # MSE for Marked Cetaceans
    fig.add_trace(error_fig.data[1], row=3, col=1)  # MSE for Hydrophone Detection
    fig.add_trace(error_fig.data[2], row=3, col=1)  # MSE for Combined
    
    # Update x and y axis labels
    fig.update_xaxes(title_text="X Position", row=1, col=1)
    fig.update_yaxes(title_text="Y Position", row=1, col=1)
    
    fig.update_xaxes(title_text="X Position", row=1, col=2)
    fig.update_yaxes(title_text="Y Position", row=1, col=2)
    
    fig.update_xaxes(title_text="X Position", row=2, col=1)
    fig.update_yaxes(title_text="Y Position", row=2, col=1)
    
    fig.update_xaxes(title_text="X Position", row=2, col=2)
    fig.update_yaxes(title_text="Y Position", row=2, col=2)
    
    # For the error plot (MSE plot)
    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    fig.update_yaxes(title_text="MSE", row=3, col=1)
        # Update the layout
    fig.update_layout(
        showlegend=False
    )
    
    return fig

def plot_error_over_time(errors_over_time):
    timesteps = np.arange(len(errors_over_time))

    if errors_over_time:
        mse_marked, mse_detected, mse_combined = zip(*errors_over_time)
    else:
        mse_marked, mse_detected, mse_combined = [], [], []  # Default empty lists if no data
        print("No errors to unpack.")
    
    # Create the Plotly figure
    fig = go.Figure()

    # Add traces for each error type
    fig.add_trace(go.Scatter(x=timesteps, y=mse_marked, mode='lines', name="Marked Cetaceans", line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=timesteps, y=mse_detected, mode='lines', name="Hydrophone Detection", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=timesteps, y=mse_combined, mode='lines', name="Marked + Hydrophone", line=dict(color='purple', dash='solid')))
    
    # Update the layout
    fig.update_layout(
        title="Estimation Error Over Time",
        xaxis_title="Time Step",
        yaxis_title="Mean Squared Error (MSE)",
        template="plotly_dark",
        showlegend=False
    )

    return fig

####




