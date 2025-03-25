import streamlit as st
import plotly.graph_objects as go
import requests
from utils.simulation import run_simulation, plot_density_and_trajectory_at_time_step  # Adjust this based on actual functions in simulation.py
st.set_page_config(layout="wide")
                       
def main():
    st.title("Cetacean Detection Simulation")
    
    # Sidebar for simulation parameters
    with st.sidebar:
        st.header("Simulation Parameters")
        N = st.number_input("Number of Cetaceans", value=50, step=1)
        M = st.number_input("Marked Cetaceans", value=5, step=1)
        correlation_strength = st.number_input("Correlation Strength", value=0.4, step=0.1)
        xlim = st.number_input("X Limit", value=400, step=10)
        ylim = st.number_input("Y Limit", value=400, step=10)
        steps = st.number_input("Steps", value=100, step=1)
        num_hydrophones = st.number_input("Number of Hydrophones", value=5, step=1)
        detection_range = st.number_input("Detection range", value=50, step=10)
        run_button = st.button("Run Simulation")
    
    if run_button:
        # Parameters to be passed for simulation
        params = {
            "N": N,
            "M": M,
            "correlation_strength": correlation_strength,
            "xlim": xlim,
            "ylim": ylim,
            "steps": steps,
            "num_hydrophones": num_hydrophones,
            "detection_range": detection_range
        }
        
        # Run simulation and capture results
        density_over_time, cetaceans, hydrophones, errors_over_time = run_simulation(**params)
        
        # Store the results in session state for later use
        st.session_state["density_over_time"] = density_over_time
        st.session_state["errors_over_time"] = errors_over_time
        st.session_state["cetaceans"] = cetaceans
        st.session_state["hydrophones"] = hydrophones

        # Store the total number of steps in session state
        st.session_state["total_steps"] = len(density_over_time)
        st.session_state["current_step"] = 0
    
    if "density_over_time" in st.session_state:
        # Retrieve the stored results from session state
        density_over_time = st.session_state["density_over_time"]
        errors_over_time = st.session_state["errors_over_time"]
        cetaceans = st.session_state["cetaceans"]
        hydrophones = st.session_state["hydrophones"]
        
        # Time step slider
        step = st.slider("Time Step", 0, st.session_state["total_steps"] - 1, st.session_state["current_step"])
        st.session_state["current_step"] = step
        
        # Call the function to plot the density and trajectory for the current step
        fig = plot_density_and_trajectory_at_time_step(
            step, 
            density_over_time, 
            errors_over_time, 
            cetaceans, 
            hydrophones, 
            xlim, 
            ylim
        )

        # Set the figure size (width and height)
        fig.update_layout(
            width=None,  # Set the width of the chart
            height=1600,  # Set the height of the chart
        )

        # Plot the figure in Streamlit with increased space
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
