import matplotlib.pyplot as plt
import numpy as np

# Load city coordinates and distance matrix (unchanged)
def load_data(city_file, distance_file):
    with open(city_file, "r") as fhand:
        coordinates = [list(map(float, line.rstrip().split()[1:])) for line in fhand]

    with open(distance_file, "r") as fhand_2:
        distances = [list(map(int, line.rstrip().split())) for line in fhand_2]

    coordinates = np.array(coordinates)  # Shape: (42, 2)
    distances = np.array(distances)      # Shape: (42, 42)
    return coordinates, distances

# Nearest Neighbor TSP
def nearest_neighbor_tsp(distance_matrix, start_city):
    num_cities = len(distance_matrix)
    visited = [False] * num_cities
    tour = [start_city]
    visited[start_city] = True
    total_distance = 0

    current_city = start_city
    for _ in range(num_cities - 1):
        # Find the nearest unvisited city
        next_city = np.argmin([
            distance_matrix[current_city, i] if not visited[i] else float('inf') 
            for i in range(num_cities)
        ])
        total_distance += distance_matrix[current_city, next_city]
        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city

    # Return to the starting city
    total_distance += distance_matrix[current_city, start_city]
    tour.append(start_city)

    return tour, total_distance

# Plot tour (unchanged)
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
    ax.set_title(f"Total Distance (Nearest neighbour): {distance}")
    ax.legend()
    ax.grid(True)


distances = set()
# Main execution
if __name__ == "__main__":
    city_file = "cityData.txt"
    distance_file = "intercityDistance.txt"
    
    coordinates, distance_matrix = load_data(city_file, distance_file)

    for i in range(42):
        # Solve TSP using Nearest Neighbor
        start_city = np.random.choice([i for i in range(0,42,1)])  # Can be any city
        route, distance = nearest_neighbor_tsp(distance_matrix, start_city)
        distances.add(distance)

    distances = sorted(distances)

    fig, axes = plt.subplots(figsize=(16,9), nrows=2, ncols=3)
    axes = axes.ravel()
    for i in range(6):
        print(f"{i+1}. Best Nearest Neighbor Route")
        print(f"Total Distance covered by the {i+1}. route (Nearest Neighbor): {distances[i]}")
        plot_tour(coordinates, route, distances[i], axes[i])
    
    plt.tight_layout(pad=2.0)  # Automatically adjusts spacing with padding
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.2, hspace=0.3)
    fig.savefig("graphs_of_nn/graph_of_nearest_neighbour_1")
    plt.show()
