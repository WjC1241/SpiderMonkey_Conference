using Agents                   # For agent-based modeling
using CSV                      # To handle CSV files
using GLMakie                  # For plotting and visualization
using DataFrames               # For data manipulation
using Dates                    # For date/time handling
using Random                   # For random number generation
using StatsBase                # For statistical operations
using CairoMakie

# Define counters to time functions
global total_move_time = 0.0
global total_feed_time = 0.0
global total_reproduce_time = 0.0
global total_die_time = 0.0
global total_increment_age_time = 0.0 
global total_move_female_time = 0.0
global total_monkey_step_time = 0.0
global total_patch_step_time = 0.0
global total_create_territory_time = 0.0

# Define the Monkey Agent
@agent struct Monkey(GridAgent{2})  # Define a struct for Monkey agents on a 2D grid
    sex::String                     # Sex of the monkey (male/female)
    age::Float64                    # Age of the monkey in years
    energy::Float64                 # Energy level of the monkey in kcal
    state::String                   # Current state of the monkey (Alive, Hungry, Dead)
    radius_movement::Int            # Movement radius of the monkey in cells
    reproduction_timer::Int         # Time left until the monkey can reproduce again (in days)
    territory::Int                  # ID to track which territory the monkey belongs to
    has_moved::Bool                 # Verify if a female monkey has changed territories
end

# Define a mutable structure for counters
mutable struct Counters
    step_count::Int                # Contador de pasos de la simulación
    starvation_deaths::Vector{Int} # Muertes por hambre por territorio
    age_deaths::Vector{Int}        # Muertes por edad por territorio
    female_counts::Vector{Int}     # Cantidad de hembras por territorio
end

# Function to select a random item from a list based on weights
function weighted_random_choice(rng::AbstractRNG, items::Vector{T}, weights::Vector{Float64}) where T   
    index = sample(rng, 1:length(items), Weights(weights))                                              # Select an index based on the weights
    return items[index]                                                                                 # Return the item at the selected index
end

# Function to configure a single territory (configures patches and agents for one territory)
function create_territory(model, territory_id::Int, patch_dim::Float64, dims_per_territory::Tuple{Int, Int}; n_monkeys::Int)
    global total_create_territory_time += @elapsed begin
        rng = model.rng                                                                 # Call the same seed for random choices
        (x_min, x_max, y_min, y_max) = model.territory_info[territory_id][:boundaries]  # Retrieve the boundaries for the current territory
        model.counters.female_counts[territory_id] = 0                                  # Initialize the counters of females

        # Read the csv patch_type files
        df = CSV.read("./data/patch_types_before.csv", DataFrame)
        type_distribution = Dict()                                                      # Create a dictionary to store the data

        # Iterate over each row of the DataFrame to built the distribution dictionary
        for row in eachrow(df)
            caloric_value = row[:KCAL_TREE]                                                                                                                         # Calculate the caloric value adjusted to the patch area
            active_months = [i for (i, val) in enumerate(row[["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]) if val == 1]    # Determine the active months evaluating each column 
            type_distribution[row[:GENUS]] = (caloric_value, active_months, row[:LAND_PERC_OG])                                                                     # Add to the dictionary 
        end

        # Initialize the grid by assigning patch types to random positions based on the type distribution
        grid_width, grid_height = dims_per_territory
        grid_size = grid_width * grid_height

        # Create a list of every possible position
        all_positions = [(x, y) for x in x_min:x_max, y in y_min:y_max]
        shuffled_positions = shuffle(copy(all_positions))                   # Shuffle all possitions to ensure randomness
        current_index = 1                                                   # Create an index to keep track of the assigned possitions

        # Loop through the type distribution to place patches on the grid
        for (_, (value, active_months, percentage)) in type_distribution    
            num_patches = Int(floor(grid_size * percentage))                # Calculate how many patches of this type to place
            for _ in 1:num_patches
                pos = shuffled_positions[current_index]                     # Obtained an unoccupied possition 
                current_index += 1                                          # Update the index
                model.fruit_value[pos...] = value                           # Set the fruit value at this position
                model.original_fruit_value[pos...] = value                  # Store the original fruit value
                model.active_months[pos...] = active_months                 # Set the active months for this patch
            end
        end

        # Initialize monkey agents for this territory
        for _ in 1:n_monkeys
            energy = rand(rng, 1040.0:2000.0)                                                                               # Randomly assign initial energy level within range
            sex = weighted_random_choice(rng, ["female", "male"], [1.5, 1.0])                                               # Randomly assign sex based on weighted choice
            age_range = weighted_random_choice(rng, [0:8, 9:20], [1.0, 1.38])                                               # Define the probabilities of each age range with weighted choice
            age = rand(rng, age_range)                                                                                      # Randomly assign age within the selected range
            state = "Alive"                                                                                                 # Set initial state to "Alive"
            reproduction_timer = rand(rng, 0:(4*360))                                                                       # Assign a random reproduction timer (in days)
            radius_movement = floor(Int, 150 / patch_dim)                                                                   # Calculate movement radius based on patch size
            pos = (rand(rng, x_min:x_max), rand(rng, y_min:y_max))                                                          # Assign a position between the boundaries of the territory

            agente = add_agent!(Monkey, model, sex, age, energy, state, radius_movement, reproduction_timer, 
            territory_id, false, pos = pos)                                                                                 # Add monkey to the model, assigning it to the current territory
            move_agent!(agente, pos, model)                                                                                 # Manually move the agent to the position between the boundaries

            # Update the counter of the female monkeys for the territory
            if sex == "female"
                model.counters.female_counts[territory_id] += 1
            end
        end
    end
end

# Function to move a monkey to the best nearby patch within its own territory based on fruit value
function move!(monkey::Monkey, model)
    global total_move_time += @elapsed begin
        territory_id = monkey.territory                                                                         # Get the current territory ID of the monkey
        (x_min, x_max, y_min, y_max) = model.territory_info[territory_id][:boundaries]                          # Get the boundaries of the monkey's assigned territory
        
        nearby_positions_list = [pos for pos in nearby_positions(monkey, model, monkey.radius_movement) if      # Filter nearby positions to only include those within the territory boundaries
                                 x_min <= pos[1] <= x_max && y_min <= pos[2] <= y_max]                          # Keep positions within territory bounds
        
        max_fruit_value = maximum(model.fruit_value[pos...] for pos in nearby_positions_list)                   # Find the maximum fruit value among nearby positions
        best_positions = [pos for pos in nearby_positions_list if model.fruit_value[pos...] == max_fruit_value] # Filter the list to only include positions with the maximum fruit value
        
        # Keep trying to select a position until an unoccupied one is found
        while !isempty(best_positions)                                                                          # Check if the selected position is not overcrowded
            selected_pos = rand(model.rng, best_positions)                                                      # Randomly select one of the best positions using the model's RNG
            if count(agent -> agent.pos == selected_pos, allagents(model)) < 25                                 # Check if there are fewer than 25 agents at the desired position
                move_agent!(monkey, selected_pos, model)                                                        # Move the monkey to the best unoccupied position
                return
            else
                best_positions = filter(pos -> pos != selected_pos, best_positions)                             # Remove the selected position from the list and try again
            end
        end
    end
end

# Function to feed the monkey based on available fruit in its movement range
function feed!(monkey::Monkey, model)
    global total_feed_time += @elapsed begin
        target_energy = 2000 - monkey.energy                                        # Set the target energy level for the monkey

        if target_energy > 0                                                        # If the monkey needs energy
            total_fruit_available = 0                                               # Initialize a counter for total available fruit
            patches_in_range = []                                                   # Initialize a list for patches within the monkey's movement range
            for pos in nearby_positions(monkey, model, monkey.radius_movement)      # Loop through all patches in range
                if model.fruit_value[pos...] > 0                                    # If there is fruit available in this patch
                    total_fruit_available += model.fruit_value[pos...]              # Add the fruit value to the total
                    push!(patches_in_range, pos)                                    # Add the patch to the list of patches with fruit
                end
            end

            if total_fruit_available > 0                                                        # If there is any fruit available
                energy_per_patch = target_energy / length(patches_in_range)                     # Calculate how much energy to gain from each patch
                for pos in patches_in_range                                                     # Loop through each patch
                    possible_energy_gain = min(model.fruit_value[pos...], energy_per_patch)     # Calculate the energy gain based on available fruit
                    monkey.energy += possible_energy_gain                                       # Add energy to the monkey
                    model.fruit_value[pos...] -= possible_energy_gain                           # Reduce the fruit value in the patch
                    target_energy -= possible_energy_gain                                       # Reduce the target energy for the monkey
                end
            end
        end
    end
end

# Function to handle monkey reproduction
function reproduce!(monkey::Monkey, model)
    global total_reproduce_time += @elapsed begin
        rng = model.rng
        if monkey.sex == "female" && monkey.age >= 8 && monkey.reproduction_timer <= 0 && monkey.energy > 1040              # If conditions are met for reproduction
            possible_mates = []                                                                                             # Initialize a list of possible mates
            for agent in nearby_agents(monkey, model, monkey.radius_movement)                                                                  # Search for male agents in the breed area
                if agent.sex == "male" && agent.age >= 8 && agent.energy > 1040 && agent.territory == monkey.territory      # If the agent is a valid mate
                    push!(possible_mates, agent)                                                                            # Add the agent to the list of possible mates
                end
            end

            if !isempty(possible_mates)
                baby_sex = weighted_random_choice(rng, ["female", "male"], [1.5, 1.0])              # Determine the sex of the new agent                                                             # If there are potential mates
                add_agent!(                                                                         # Create a new monkey agent (baby) with random properties
                    Monkey, model,
                    sex = baby_sex,                                                                 # Randomly assign sex (60% female)
                    age = 0,                                                                        # Set age to 0 (newborn)
                    energy = rand(rng, 1040.0:2000.0),                                              # Assign random initial energy
                    state = "Alive",                                                                # Set state to "Alive"
                    radius_movement = monkey.radius_movement,                                       # Inherit movement radius from mother
                    reproduction_timer = 8 * 360,                                                   # Assign reproduction timer to 8 years
                    territory = monkey.territory,
                    has_moved = false,
                    pos = monkey.pos                                                                # Place the newborn in the same position as the mother
                )

                # Update the counter for female monkeys in the territory
                if baby_sex == "female"
                    model.counters.female_counts[monkey.territory] += 1
                end

                monkey.reproduction_timer = 4 * 360                                                 # Reset the mother's reproduction timer (4 years)
            end
        end
    end
end

# Function to handle death of a monkey
function maybe_die!(monkey::Monkey, model)
    global total_die_time += @elapsed begin
        if monkey.energy <= 0                                           # If the monkey has no energy, it dies of starvation
            monkey.state = "Dead by starvation"                         # Update monkey state
            model.counters.starvation_deaths[monkey.territory] += 1     # Increment the starvation death counter
            
            # Update gender counter
            if monkey.sex == "female"
                model.counters.female_counts[monkey.territory] -= 1     # Decrement the female counter if female
            end
            remove_agent!(monkey, model)                                # Remove the monkey from the model

        elseif monkey.age >= 25                                         # If the monkey reaches age 25, it dies of old age
            monkey.state = "Dead by aging"                              # Update monkey state
            model.counters.age_deaths[monkey.territory] += 1            # Increment the age death counter

            # Update gender counter
            if monkey.sex == "female"
                model.counters.female_counts[monkey.territory] -= 1     # Decrement the female counter if female
            end
            remove_agent!(monkey, model)                                # Remove the monkey from the model
        end
    end
end

# Function to increment the age of a monkey and decrease its energy
function increment_age!(monkey::Monkey, model)
    global total_increment_age_time += @elapsed begin
        monkey.age += 1 / 360                           # Increase the monkey's age by one day (1/365 year)
        monkey.energy -= 464                            # Decrease the monkey's energy by 464 kcal (daily energy consumption)
        if monkey.reproduction_timer > 0                # If the reproduction timer is active
            monkey.reproduction_timer -= 1              # Decrease the reproduction timer by 1 day
        end
    end
end

# Move female monkey between territories
function move_female_between_territories!(monkey::Monkey, model)
    global total_move_female_time += @elapsed begin
        territory_info = model.territory_info 
        rng = model.rng
        current_territory = monkey.territory                                        # Get the current territory ID of the monkey
        neighbors = territory_info[current_territory][:neighbors]                   # Retrieve the list of neighboring territories for the current territory

        # Look for territories with fewer females
        best_territory = current_territory                                          # Initialize best_territory as the current territory
        current_females = model.counters.female_counts[current_territory]           # Get the number of females in the current territory
        for neighbor_territory in neighbors                                         # Iterate over each neighboring territory
            neighbor_females = model.counters.female_counts[neighbor_territory]     # Get the number of females in the neighboring territory
            if neighbor_females < current_females                                   # If this neighbor has fewer females than the current territor
                best_territory = neighbor_territory                                 # Update best_territory to the neighboring territory with fewer females
            end
        end

        # Move the monkey to the new territory if a better one is found
        if best_territory != current_territory
            monkey.territory = best_territory                                           # Update monkey's territory
            (x_min, x_max, y_min, y_max) = territory_info[best_territory][:boundaries]  # Retrieve the boundary limits for the new territory
            new_pos = (rand(rng, x_min:x_max), rand(rng, y_min:y_max))                  # Select a random position within the new territory's boundaries
            move_agent!(monkey, new_pos, model)                                         # Move the monkey to the new position on the grid
            
            # Update female counts
            model.counters.female_counts[current_territory] -= 1
            model.counters.female_counts[best_territory] += 1

            # Marked the agent as has_moved
            monkey.has_moved = true  # Marca el agente como movido
        else
            # If staying in the current territory, move like any other monkey
            move!(monkey, model)
        end
    end
end

# Function defining Monkey agent behavior
function monkey_step!(monkey::Monkey, model)
    global total_monkey_step_time += @elapsed begin
        # Handle female behavior between territories
        if monkey.sex == "female" && monkey.age >= 8 && !monkey.has_moved                                                                                                                           # If it leaves
            move_female_between_territories!(monkey, model)                 # Move to another territory
        else
            move!(monkey, model)                                            # Normal movement for other monkeys
        end
        feed!(monkey, model)                                                # Call the feeding function for the monkey
        reproduce!(monkey, model)                                           # Call the reproduction function for the monkey
        maybe_die!(monkey, model)                                           # Check if the monkey dies (from starvation or old age)
        increment_age!(monkey, model)                                       # Increment the monkey's age and decrease energy
    end
end

function deforest_patches!(model, percentage::Float64)
    for territory_id in 1:length(model.territory_info)
        (x_min, x_max, y_min, y_max) = model.territory_info[territory_id][:boundaries]
        # Obtener todas las posiciones dentro del territorio
        territory_positions = [(x, y) for x in x_min:x_max, y in y_min:y_max]
        
        # Encontrar los valores únicos de `original_fruit_value` en las posiciones del territorio
        unique_fruit_values = unique(model.original_fruit_value[pos...] for pos in territory_positions if model.original_fruit_value[pos...] > 0)
        
        # Para cada tipo de patch, seleccionar aleatoriamente el porcentaje a eliminar
        for fruit_value in unique_fruit_values
            # Filtrar las posiciones que corresponden al valor actual
            positions_of_type = filter(pos -> model.original_fruit_value[pos...] == fruit_value, territory_positions)
            
            # Calcular el número de parches a talar
            num_patches_to_deforest = Int(floor(length(positions_of_type) * percentage))
            
            # Seleccionar aleatoriamente las posiciones a talar
            positions_to_deforest = shuffle(positions_of_type)[1:num_patches_to_deforest]
            
            # Actualizar los valores de los parches seleccionados
            for pos in positions_to_deforest
                model.fruit_value[pos...] = 0.0
                model.original_fruit_value[pos...] = 0.0
                model.active_months[pos...] = Int[]
            end
        end
    end
end

# Function defining the behavior of forest patches during a simulation step
function patch_step!(model::StandardABM)
    global total_patch_step_time += @elapsed begin
        model.counters.step_count += 1
        current_month = (model.counters.step_count ÷ 30) % 12 + 1

        # Función de la tala de arboles
        if model.counters.step_count % 360 == 0
            deforest_patches!(model, 0.018) # Tala del 1.8% distribuida equitativamente
        end

        # Actualizar los valores de los parches activos
        @inbounds for pos in positions(model)
            if current_month in model.active_months[pos...]
                model.fruit_value[pos...] = model.original_fruit_value[pos...]
            else
                model.fruit_value[pos...] += 0
            end
        end

        # Imprimir información de energía total de los parches
        if model.counters.step_count % 30 == 0
            alive_monkeys = count(monkey -> monkey.state == "Alive", allagents(model))
            total_patch_energy = sum(model.fruit_value)
            println("Step: ", model.counters.step_count, " - Month: ", current_month, 
                    " - Alive monkeys: ", alive_monkeys, " - Total patch energy: ", total_patch_energy)
        end
    end
end

# Higher-level function to organize multiple territories in a matrix layout
function run_multiple_territories(n_territories::Int, grid_layout::Tuple{Int, Int}; steps::Int, n_monkeys_per_territory::Int, patch_dim::Float64, tot_dim::Int, seed::Int)
    # Dimensions of each territory
    dims_per_territory = (floor(Int, tot_dim / patch_dim), floor(Int, tot_dim / patch_dim))
    
    # Calculate the total size of the grid of territories
    total_dims = (dims_per_territory[1] * grid_layout[1], dims_per_territory[2] * grid_layout[2])
    space = GridSpace(total_dims, periodic=false)

    # Define model properties: fruit values, active months, and counters for tracking events
    properties = (
        fruit_value = zeros(Float64, total_dims),  
        original_fruit_value = zeros(Float64, total_dims),
        active_months = fill(Int[], total_dims),
        percentage = zeros(Float64, total_dims),
        counters = Counters(0, zeros(Int, n_territories), zeros(Int, n_territories), zeros(Int, n_territories)),
        territory_info = Dict{Int, Dict{Symbol, Any}}(),
        rng = MersenneTwister(seed) 
    )
    
    # Initialize the agent-based model with monkey agents
    model = StandardABM(Monkey, space; 
        agent_step! = monkey_step!, model_step! = patch_step!,         # Define agent and model step functions
        properties, scheduler = Schedulers.Randomly(), warn = false    # Set model properties, RNG, and scheduling
    )
    
    # Loop through grid to create and position territories in a matrix layout
    for row in 1:grid_layout[1]
        for col in 1:grid_layout[2]
            territory_id = (row - 1) * grid_layout[2] + col        # Calculate unique territory ID based on row and column position
            
            # Calculate the spatial limits (boundaries) for each territory
            x_min = max((row - 1) * dims_per_territory[1], 1)       # Calculate minimum x-boundary (starting from 1)
            x_max = row * dims_per_territory[1]                     # Calculate maximum x-boundary for the current territory
            y_min = max((col - 1) * dims_per_territory[2], 1)       # Calculate minimum y-boundary (starting from 1)
            y_max = col * dims_per_territory[2]                     # Calculate maximum y-boundary for the current territory

            # Store the boundaries of the current territory
            boundaries = (x_min, x_max, y_min, y_max)

            # Calculate the neighbors (territories surrounding this one in the grid)
            neighbors = []                                         # Initialize an empty list to store neighboring territories
            for row_offset in -1:1                                 # Iterate over row offsets (-1, 0, 1) to check adjacent rows
                for col_offset in -1:1                             # Iterate over column offsets (-1, 0, 1) to check adjacent columns
                    neighbor_row = row + row_offset                # Calculate neighboring row
                    neighbor_col = col + col_offset                # Calculate neighboring column
                    
                    # Ensure the neighboring row and column are within grid bounds and are not the current territory
                    if 1 <= neighbor_row <= grid_layout[1] && 1 <= neighbor_col <= grid_layout[2] && !(row_offset == 0 && col_offset == 0)
                        neighbor_territory_id = (neighbor_row - 1) * grid_layout[2] + neighbor_col       # Calculate the ID of the neighboring territory
                        push!(neighbors, neighbor_territory_id)                                          # Add the neighboring territory ID to the list of neighbors
                    end
                end
            end

            # Store the territory info in the dictionary
            model.territory_info[territory_id] = Dict(
                :boundaries => boundaries,  # Save the boundaries of the territory
                :neighbors => neighbors     # Save the list of neighboring territories
            )
            
            # Create the territory
            create_territory(model, territory_id, patch_dim, dims_per_territory; n_monkeys=n_monkeys_per_territory)
        end
    end

    # Define mdata for visualization
    mdata = [
        model -> count(agent -> agent.sex == "female", allagents(model)),
        model -> count(agent -> agent.sex == "male", allagents(model)),
        model -> count(agent -> agent.state == "Alive", allagents(model)),
        model -> sum(copy(model.counters.starvation_deaths))
    ]

    # Visualización interactiva con abmexploration
    fig, abmobs = abmexploration(
        model;
        agent_color = (monkey -> monkey.sex == "female" ? "#ff69b4" : "#4169e1"),
        agent_marker = (monkey -> monkey.age < 8 ? :circle : :diamond),
        agent_size = 8,
        heatarray = :fruit_value,
        heatkwargs = (colormap = :Greens, colorrange = (0, maximum(model.fruit_value))),
        add_colorbar = true,
        mdata = mdata,
        mlabels = ["Female monkeys", "Male Monkeys", "Alive Monkeys", "Dead Monkey by Starvation"],
        controls = true
    )

    return fig, model  # Return the final model state with data collected
end

# Función para grabar la simulación en video
function record_monkey_simulation!(model::ABM; filename="monkeys_simulation.mp4", steps=360, fps=10)
    abmvideo(
        filename, model;
        frames=steps,
        framerate=fps,
        title="Monkey Population Simulation",
        agent_color = (monkey -> monkey.sex == "female" ? "#ff69b4" : "#4169e1"),
        agent_marker = (monkey -> monkey.age < 8 ? :circle : :diamond),
        agent_size = 8,
        heatarray = :fruit_value,
        heatkwargs = (colormap = :Greens, colorrange = (0, maximum(model.fruit_value))),
        add_colorbar = false,
        dt = 1
    )
    println("✅ Video guardado en: $filename")
end

# Extraer solo el modelo de la tupla que devuelve `run_multiple_territories`
fig, model = run_multiple_territories(2, (1, 2), steps=360, n_monkeys_per_territory=25, patch_dim=3.7, tot_dim=2000, seed=23180)

# Ejecutar la simulación y guardar el video
record_monkey_simulation!(model, filename="monkeys_simulation.mp4", steps=360, fps=10)

# Run the simulation
#fig, model = run_multiple_territories(2, (1, 2), steps=36, n_monkeys_per_territory=25, patch_dim=3.7, tot_dim=2000, seed=23180)
#display(fig)
