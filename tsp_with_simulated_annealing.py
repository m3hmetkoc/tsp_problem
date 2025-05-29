import numpy as np
import matplotlib.pyplot as plt
import random

# Load city coordinates and distance matrix (unchanged)
def load_data(city_file, distance_file):
    with open(city_file, "r") as fhand:
        coordinates = [list(map(float, line.rstrip().split()[1:])) for line in fhand]

    with open(distance_file, "r") as fhand_2:
        distances = [list(map(int, line.rstrip().split())) for line in fhand_2]

    coordinates = np.array(coordinates)  # Shape: (42, 2)
    distances = np.array(distances)      # Shape: (42, 42)
    return coordinates, distances

# Calculate the total distance of a given route
def calculate_total_distance(route, distance_matrix):
    total_distance = sum(
        distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)
    )
    total_distance += distance_matrix[route[-1], route[0]]  # Return to start
    return total_distance

# Simulated Annealing TSP
def simulated_annealing_tsp(distance_matrix, initial_temperature=1000, cooling_rate=0.9985, max_iterations=10000):
    num_cities = len(distance_matrix)
    current_route = list(range(num_cities))
    random.shuffle(current_route)  # Start with a random route 
    current_distance = calculate_total_distance(current_route, distance_matrix)

    best_route = current_route[:]
    best_distance = current_distance

    temperature = initial_temperature

    for _ in range(max_iterations):
        # Generate a new candidate route by swapping two cities
        i, j = random.sample(range(num_cities), 2)
        new_route = current_route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        
        new_distance = calculate_total_distance(new_route, distance_matrix)
        
        # Decide whether to accept the new route
        if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / temperature):
            current_route = new_route
            current_distance = new_distance

            # Update the best route found so far
            if new_distance < best_distance:
                best_route = new_route[:]
                best_distance = new_distance

        # Cool down the temperature
        temperature *= cooling_rate
        # Terminate early if the temperature is very low
        if temperature < 1e-9:
            break
    return best_route, best_distance

# Plot tour (unchanged)
# Updated plot_tour function
def plot_tour(coordinates, route, distance, ax):
    # Append the starting city to the end of the route for a closed loop
    route_coordinates = coordinates[route + [route[0]]]
    ax.plot(route_coordinates[:, 0], route_coordinates[:, 1], '-o', label="Path", alpha=0.6)
    
    # Draw scatter points with larger size and lower alpha
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=350, c='blue', alpha=0.4, label="Cities")
    
    # Annotate each city with its ID inside the scatter marker
    for city_id, (x, y) in enumerate(coordinates):
        ax.text(x, y, str(city_id), fontsize=10, color='white', ha='center', va='center', fontweight='bold')
    
    # Annotate the starting city specially
    start_x, start_y = coordinates[route[0]]
    ax.text(start_x, start_y-3, 'Start', fontsize=9, color='green', fontweight='bold', ha='center', va='center')

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Total Distance (Simulated Annealing): {distance}")
    ax.legend()
    ax.grid(True)

distances = set()
# Main execution
if __name__ == "__main__":
    city_file = "cityData.txt"
    distance_file = "intercityDistance.txt"
    coordinates, distance_matrix = load_data(city_file, distance_file)
# Solve TSP using Simulated Annealing
    for i in range(50):
    # Solve TSP using Nearest Neighbor
        best_route, best_distance = simulated_annealing_tsp(distance_matrix)
        distances.add(best_distance)
    
    distances = sorted(distances)

    fig, axes = plt.subplots(figsize=(16, 8), nrows=2, ncols=3)  # Adjust layout automatically
    axes = axes.ravel()

    for i in range(6):
        print(f"{i+1}. best Route (Simulated Annealing)")
        print(f"Total Distance covered by the {i+1}. route (Simulated Annealing): {distances[i]}")
        plot_tour(coordinates, best_route, distances[i], axes[i])

    # Adjust spacing around the subplots
        # Adjust spacing and margins
    plt.tight_layout(pad=2.0)  # Automatically adjusts spacing with padding
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.2, hspace=0.3)
    fig.savefig("graphs_of_sa/graph_of_simulated_annealing_3")
    plt.show()

