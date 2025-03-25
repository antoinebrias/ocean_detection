# Cetacean Detection Simulation

This repository contains a simulation framework designed to study cetacean movement patterns and their detection through hydrophone networks. By simulating cetacean movements, hydrophone placements, and detection algorithms, this project aims to improve monitoring and conservation strategies for marine species.

The simulation generates real-time data visualizations to analyze the behavior of marked and unmarked cetaceans over time, providing insight into their population dynamics and detection accuracy.

You can try the app here : https://antoinebrias-ocean-detection-streamlit-main-uzzcfa.streamlit.app/

## Features

- **Simulated Cetacean Movement**: A dynamic model for simulating cetacean behavior in a given habitat, taking into account various parameters like movement correlation strength, population size, and environmental constraints.
- **Hydrophone Detection**: The simulation includes hydrophones placed at strategic locations, detecting cetacean positions based on a set of predefined rules.
- **Real-Time Visualization**: Utilizes `Plotly` and `Streamlit` to create interactive visualizations, including heatmaps and trajectory plots, to represent cetacean density and movement patterns.
- **Error Metrics**: Implements Mean Squared Error (MSE) calculations to assess the accuracy of the detection process, comparing the simulated detection data to the actual positions.
- **Interactive Interface**: Streamlit is used for creating an interactive dashboard, where users can adjust parameters and see the results in real-time.

## Installation

To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cetacean-detection-simulation.git
    cd cetacean-detection-simulation
    ```

2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows, use venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the simulation with Streamlit:
    ```bash
    streamlit run streamlit_main.py
    ```

## Usage

Once the simulation is running, you'll be able to interact with the interface to:

- **Adjust Parameters**: Modify variables like the number of cetaceans, hydrophone count, correlation strength, etc.
- **View Results**: Explore the density heatmaps and trajectory plots of cetacean movements and hydrophone detections.
- **Error Analysis**: Visualize the Mean Squared Error (MSE) values to assess the detection accuracy of marked and unmarked cetaceans.

## File Structure

cetacean-detection-simulation/ 
│ 

├── simulation.py # Main simulation logic 

├── streamlit_main.py # Streamlit dashboard to run the simulation 

├── utils/ 

│ └── simulation.py # Helper functions for simulation 

├── requirements.txt # Project dependencies 

└── README.md # Project overview and instructions

## Dependencies

- `numpy`
- `matplotlib`
- `scipy`
- `ipywidgets`
- `plotly`
- `streamlit`

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to open issues and pull requests for any improvements or bug fixes. Contributions are welcome!

## License

This project is licensed under the MIT License.
