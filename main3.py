import requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch equivalent production hours from PVGIS API
def get_equivalent_production_hours(lat, lon, installed_capacity_kw, system_loss, startyear=2018, endyear=2020):
    # Define the parameters for the API request
    params = {
        'lat': lat,
        'lon': lon,
        'outputformat': 'json',
        'startyear': startyear,
        'endyear': endyear,
    }

    # PVGIS API URL
    url = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc'

    # Send the request to PVGIS API with a timeout
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from PVGIS API: {e}")
        return None

    data = response.json()
    if 'outputs' in data and 'hourly' in data['outputs']:
        hourly_data = data['outputs']['hourly']

        # Calculate total irradiation in kWh/m² over the period
        total_years = endyear - startyear + 1
        equivalent_hours = sum(
            min(item['G(i)'] / 1000, 1) if item['G(i)'] / 1000 > 1 else item['G(i)'] / 1000 for item in hourly_data
        ) * (1 - system_loss / 100) / total_years

        return {
            "equivalent_production_hours": equivalent_hours,
            "hourly_data": hourly_data
        }
    else:
        st.error("Unexpected response format from PVGIS API.")
        return None

# Function to generate a fictional load curve adjusted to match total yearly consumption
def generate_fictional_load_curve(hours, total_yearly_consumption):
    # Generate a realistic load curve for a company operating 7 days a week
    # Load pattern: higher during working hours (6 AM - 8 PM), lower at night

    load_curve = []
    for hour in range(hours):
        hour_of_day = hour % 24

        # Base load during off-hours
        base_load = 50  # kW

        # Increase during working hours (6 AM to 8 PM)
        if 6 <= hour_of_day < 20:
            load = base_load + np.random.uniform(100, 200)  # Peak load between 150-250 kW
        else:
            load = base_load + np.random.uniform(0, 20)  # Off-peak load between 50-70 kW

        load_curve.append(load)

    # Convert load_curve to numpy array for easier calculations
    load_curve = np.array(load_curve)

    # Calculate the total consumption of the generated load curve
    total_consumption = sum(load_curve)  # kWh

    # Scale the load curve to match the total yearly consumption specified by the user
    scaling_factor = (total_yearly_consumption) / total_consumption

    # Apply scaling
    adjusted_load_curve = load_curve * scaling_factor

    # Recalculate total consumption to check the error margin
    adjusted_total_consumption = sum(adjusted_load_curve)

    # Calculate the percentage error
    error_percentage = abs((adjusted_total_consumption - total_yearly_consumption) / total_yearly_consumption) * 100

    # Ensure the error is within the acceptable margin (1-2%)
    if error_percentage > 2:
        st.warning(f"The adjusted load curve deviates from the desired total consumption by {error_percentage:.2f}%.")
    else:
        st.success(
            f"The adjusted load curve matches the desired total consumption within {error_percentage:.2f}% error margin.")

    return adjusted_load_curve.tolist()

# Function to optimize the solar and battery system with adaptive step sizes
def optimize_solar_battery_system(load_curve, hourly_data, project_lifetime, wp_per_m2,
                                  system_loss, start_year, end_year, target_coverage,
                                  panel_cost_per_mwp, battery_cost_per_mwh, installation_cost_per_mwp,
                                  maintenance_cost_per_mwp,
                                  maintenance_cost_per_mwh_battery, battery_lifespan,
                                  panel_degradation_rate, min_state_of_charge, electricity_price, sell_price, progress_bar):
    configurations = []

    total_years = end_year - start_year + 1

    # Total energy consumption per year (kWh)
    total_consumption_per_year = sum(load_curve) / total_years

    # Total energy consumption over the project lifetime
    total_consumption_lifetime = total_consumption_per_year * project_lifetime

    # Precompute irradiation array
    irradiation_array = np.array([hour['G(i)'] / 1000 for hour in hourly_data])

    # Calculate average load (kW)
    total_hours = len(load_curve)
    average_load_kw = total_consumption_per_year / (total_hours / total_years)

    # Maximum Solar Capacity (MWp)
    max_solar_capacity_mwp = max((average_load_kw / 1000) * 10, 0.1)  # Scale factor of 10

    # Coarse Step Size for Panels (MWp) based on total yearly consumption
    coarse_step_size_mwp = max_solar_capacity_mwp / 10  # 10% of max capacity
    coarse_step_size_mwp = max(coarse_step_size_mwp, 0.1)  # Ensure a minimum step size

    # Average Daily Consumption (kWh)
    average_daily_consumption_kwh = total_consumption_per_year / 365

    # Maximum Battery Capacity (MWh) to cover 2 days of consumption
    max_battery_capacity_mwh = (average_daily_consumption_kwh * 2) / 1000  # Convert kWh to MWh

    # Coarse Step Size for Batteries (MWh) based on total yearly consumption
    coarse_step_size_mwh = max_battery_capacity_mwh / 10  # 10% of max capacity
    coarse_step_size_mwh = max(coarse_step_size_mwh, 0.1)  # Ensure a minimum step size

    # Generate arrays for coarse search
    panel_sizes_mwp_coarse = np.arange(0.0, max_solar_capacity_mwp + coarse_step_size_mwp, coarse_step_size_mwp)
    battery_sizes_mwh_coarse = np.arange(0.0, max_battery_capacity_mwh + coarse_step_size_mwh, coarse_step_size_mwh)

    total_iterations = len(panel_sizes_mwp_coarse) * len(battery_sizes_mwh_coarse)
    current_iteration = 0

    st.write("🔍 **Phase 1: Coarse Search**")
    for panels_mwp in panel_sizes_mwp_coarse:
        panel_area_m2 = (panels_mwp * 1e6) / wp_per_m2  # Convert MWp to m²
        module_efficiency = wp_per_m2 / 1000  # Convert Wp/m² to kW/m²

        # Precompute solar production for all hours
        solar_production = (panel_area_m2 * irradiation_array * module_efficiency) * (1 - system_loss / 100)

        for batteries_mwh in battery_sizes_mwh_coarse:
            # Define min and max battery states
            min_battery_state = batteries_mwh * 1000 * min_state_of_charge  # kWh
            max_battery_state = batteries_mwh * 1000  # kWh

            battery_state = max_battery_state  # Start at full charge

            covered_by_panels = 0
            covered_by_batteries = 0
            not_covered = 0

            # Initialize total excess energy for this configuration
            total_excess_energy_kwh = 0

            for hour in range(len(load_curve)):
                production = solar_production[hour]
                consumption = load_curve[hour]

                if production >= consumption:
                    surplus = production - consumption

                    # Charge the battery with surplus energy
                    available_storage = max_battery_state - battery_state
                    energy_to_store = min(surplus, available_storage)
                    battery_state += energy_to_store

                    covered_by_panels += consumption

                    # Sell excess energy to the grid if any
                    excess_energy = surplus - energy_to_store
                    if excess_energy > 0:
                        total_excess_energy_kwh += excess_energy
                else:
                    deficit = consumption - production

                    # Discharge the battery to cover the deficit
                    available_energy = battery_state - min_battery_state
                    energy_from_battery = min(available_energy, deficit)
                    battery_state -= energy_from_battery
                    covered_by_batteries += energy_from_battery

                    uncovered_energy = deficit - energy_from_battery
                    not_covered += uncovered_energy

                    covered_by_panels += production

                # Check if battery state exceeds capacity or goes below min SOC
                if battery_state > max_battery_state + 1e-6 or battery_state < min_battery_state - 1e-6:
                    break  # Invalid battery state, skip to next configuration

            else:
                # Completed all hours without invalid battery state
                total_consumption = sum(load_curve)
                portion_covered = ((covered_by_panels + covered_by_batteries) / total_consumption) * 100

                # Check if the configuration meets the target coverage
                if portion_covered < target_coverage:
                    continue  # Skip configurations that don't meet the target coverage

                # Calculate the total energy used per year
                total_used_energy_per_year = (covered_by_panels + covered_by_batteries) / total_years

                # Adjust total used energy for the project lifetime and degradation
                degradation_factors = [(1 - panel_degradation_rate) ** year for year in range(project_lifetime)]
                lifetime_used_energy_kwh = sum(total_used_energy_per_year * df for df in degradation_factors)

                # Calculate energy not covered over the project lifetime
                lifetime_uncovered_energy_kwh = total_consumption_lifetime - lifetime_used_energy_kwh

                # Calculate number of battery replacements
                number_of_battery_replacements = max(0, int((project_lifetime - 1) // battery_lifespan))

                # Calculate replacement costs
                battery_replacement_cost = number_of_battery_replacements * (batteries_mwh * battery_cost_per_mwh)

                # Calculate initial CAPEX components
                panel_cost = panels_mwp * panel_cost_per_mwp
                battery_cost = batteries_mwh * battery_cost_per_mwh
                installation_cost = panels_mwp * installation_cost_per_mwp

                # Total initial CAPEX
                initial_capex = panel_cost + battery_cost + installation_cost

                total_capex = initial_capex + battery_replacement_cost

                # Calculate OPEX components
                panel_maintenance_cost = maintenance_cost_per_mwp * panels_mwp * project_lifetime
                battery_maintenance_cost = maintenance_cost_per_mwh_battery * batteries_mwh * project_lifetime

                opex = panel_maintenance_cost + battery_maintenance_cost

                total_cost = total_capex + opex

                # Calculate revenue from energy sales
                revenue_from_sales = total_excess_energy_kwh * sell_price

                # Update net savings by adding revenue from sales
                net_savings = (total_consumption_lifetime * electricity_price) - (total_cost + (lifetime_uncovered_energy_kwh * electricity_price)) + revenue_from_sales

                # Calculate TCOE based on used energy (subtract revenue)
                tcoe = (total_cost - revenue_from_sales) / lifetime_used_energy_kwh if lifetime_used_energy_kwh > 0 else float('inf')

                # Calculate grid energy costs without solar installation
                grid_cost_without_solar = total_consumption_lifetime * electricity_price

                # Calculate remaining grid energy costs with solar installation
                grid_cost_with_solar = lifetime_uncovered_energy_kwh * electricity_price

                # Calculate total cost with solar installation (CAPEX + OPEX + remaining grid energy costs)
                total_cost_with_solar = total_cost + grid_cost_with_solar

                # Recalculate net savings with revenue
                net_savings = grid_cost_without_solar - (total_cost_with_solar - revenue_from_sales)

                # Determine if the installation saves money
                saves_money = net_savings > 0

                # Store the configuration
                config = {
                    'panels_mwp': panels_mwp,
                    'batteries_mwh': batteries_mwh,
                    'portion_covered': portion_covered,
                    'portion_not_covered': 100 - portion_covered,
                    'total_cost': total_cost,
                    'total_capex': total_capex,
                    'opex': opex,
                    'project_lifetime': project_lifetime,
                    'tcoe': tcoe,
                    'net_savings': net_savings,
                    'revenue_from_sales': revenue_from_sales,
                    'saves_money': saves_money,
                    'cost_breakdown': {
                        'Panel Cost': panel_cost,
                        'Battery Cost': battery_cost,
                        'Installation Cost': installation_cost,
                        'Battery Replacement Cost': battery_replacement_cost,
                        'Panel Maintenance Cost': panel_maintenance_cost,
                        'Battery Maintenance Cost': battery_maintenance_cost,
                        'Remaining Grid Energy Cost': grid_cost_with_solar,
                        'Revenue from Energy Sales': revenue_from_sales,
                    }
                }
                configurations.append(config)

            # Update progress bar in every iteration
            current_iteration += 1
            progress_bar.progress(min(current_iteration / total_iterations, 1.0))

    progress_bar.progress(1.0)

    # Phase 2: Fine Search
    if configurations:
        # Sort configurations by TCOE
        sorted_configurations = sorted(configurations, key=lambda x: x['tcoe'])

        # Select the top N configurations for fine search
        top_N = 3  # Number of top configurations to explore in fine search
        top_configurations_coarse = sorted_configurations[:top_N]

        st.write("🔍 **Phase 2: Fine Search**")
        fine_configurations = []

        # Fine Step Sizes
        fine_step_size_mwp = coarse_step_size_mwp / 10
        fine_step_size_mwh = coarse_step_size_mwh / 10

        # Ensure fine step sizes are not zero
        fine_step_size_mwp = max(fine_step_size_mwp, 0.01)
        fine_step_size_mwh = max(fine_step_size_mwh, 0.01)

        # Calculate total iterations for fine search
        total_iterations_fine = 0
        for config in top_configurations_coarse:
            # Define fine search ranges around the top configurations
            panels_mwp_center = config['panels_mwp']
            batteries_mwh_center = config['batteries_mwh']

            # Define fine search ranges
            panels_mwp_min = max(0.0, panels_mwp_center - coarse_step_size_mwp)
            panels_mwp_max = panels_mwp_center + coarse_step_size_mwp
            battery_mwh_min = max(0.0, batteries_mwh_center - coarse_step_size_mwh)
            battery_mwh_max = batteries_mwh_center + coarse_step_size_mwh

            # Generate fine search arrays
            panel_sizes_mwp_fine = np.arange(panels_mwp_min, panels_mwp_max + fine_step_size_mwp, fine_step_size_mwp)
            battery_sizes_mwh_fine = np.arange(battery_mwh_min, battery_mwh_max + fine_step_size_mwh, fine_step_size_mwh)

            total_iterations_fine += len(panel_sizes_mwp_fine) * len(battery_sizes_mwh_fine)

        current_iteration_fine = 0
        progress_bar_fine = st.progress(0)

        for config in top_configurations_coarse:
            panels_mwp_center = config['panels_mwp']
            batteries_mwh_center = config['batteries_mwh']

            # Define fine search ranges
            panels_mwp_min = max(0.0, panels_mwp_center - coarse_step_size_mwp)
            panels_mwp_max = panels_mwp_center + coarse_step_size_mwp
            battery_mwh_min = max(0.0, batteries_mwh_center - coarse_step_size_mwh)
            battery_mwh_max = batteries_mwh_center + coarse_step_size_mwh

            # Generate fine search arrays
            panel_sizes_mwp_fine = np.arange(panels_mwp_min, panels_mwp_max + fine_step_size_mwp, fine_step_size_mwp)
            battery_sizes_mwh_fine = np.arange(battery_mwh_min, battery_mwh_max + fine_step_size_mwh, fine_step_size_mwh)

            for panels_mwp in panel_sizes_mwp_fine:
                panel_area_m2 = (panels_mwp * 1e6) / wp_per_m2  # Convert MWp to m²
                module_efficiency = wp_per_m2 / 1000  # Convert Wp/m² to kW/m²

                # Precompute solar production for all hours
                solar_production = (panel_area_m2 * irradiation_array * module_efficiency) * (1 - system_loss / 100)

                for batteries_mwh in battery_sizes_mwh_fine:
                    # Define min and max battery states
                    min_battery_state = batteries_mwh * 1000 * min_state_of_charge  # kWh
                    max_battery_state = batteries_mwh * 1000  # kWh

                    battery_state = max_battery_state  # Start at full charge

                    covered_by_panels = 0
                    covered_by_batteries = 0
                    not_covered = 0

                    # Initialize total excess energy for this configuration
                    total_excess_energy_kwh = 0

                    for hour in range(len(load_curve)):
                        production = solar_production[hour]
                        consumption = load_curve[hour]

                        if production >= consumption:
                            surplus = production - consumption

                            # Charge the battery with surplus energy
                            available_storage = max_battery_state - battery_state
                            energy_to_store = min(surplus, available_storage)
                            battery_state += energy_to_store

                            covered_by_panels += consumption

                            # Sell excess energy to the grid if any
                            excess_energy = surplus - energy_to_store
                            if excess_energy > 0:
                                total_excess_energy_kwh += excess_energy
                        else:
                            deficit = consumption - production

                            # Discharge the battery to cover the deficit
                            available_energy = battery_state - min_battery_state
                            energy_from_battery = min(available_energy, deficit)
                            battery_state -= energy_from_battery
                            covered_by_batteries += energy_from_battery

                            uncovered_energy = deficit - energy_from_battery
                            not_covered += uncovered_energy

                            covered_by_panels += production

                        # Check if battery state exceeds capacity or goes below min SOC
                        if battery_state > max_battery_state + 1e-6 or battery_state < min_battery_state - 1e-6:
                            break  # Invalid battery state, skip to next configuration

                    else:
                        # Completed all hours without invalid battery state
                        total_consumption = sum(load_curve)
                        portion_covered = ((covered_by_panels + covered_by_batteries) / total_consumption) * 100

                        # Check if the configuration meets the target coverage
                        if portion_covered < target_coverage:
                            continue  # Skip configurations that don't meet the target coverage

                        # Calculate the total energy used per year
                        total_used_energy_per_year = (covered_by_panels + covered_by_batteries) / total_years

                        # Adjust total used energy for the project lifetime and degradation
                        degradation_factors = [(1 - panel_degradation_rate) ** year for year in range(project_lifetime)]
                        lifetime_used_energy_kwh = sum(total_used_energy_per_year * df for df in degradation_factors)

                        # Calculate energy not covered over the project lifetime
                        lifetime_uncovered_energy_kwh = total_consumption_lifetime - lifetime_used_energy_kwh

                        # Calculate number of battery replacements
                        number_of_battery_replacements = max(0, int((project_lifetime - 1) // battery_lifespan))

                        # Calculate replacement costs
                        battery_replacement_cost = number_of_battery_replacements * (batteries_mwh * battery_cost_per_mwh)

                        # Calculate initial CAPEX components
                        panel_cost = panels_mwp * panel_cost_per_mwp
                        battery_cost = batteries_mwh * battery_cost_per_mwh
                        installation_cost = panels_mwp * installation_cost_per_mwp

                        # Total initial CAPEX
                        initial_capex = panel_cost + battery_cost + installation_cost

                        total_capex = initial_capex + battery_replacement_cost

                        # Calculate OPEX components
                        panel_maintenance_cost = maintenance_cost_per_mwp * panels_mwp * project_lifetime
                        battery_maintenance_cost = maintenance_cost_per_mwh_battery * batteries_mwh * project_lifetime

                        opex = panel_maintenance_cost + battery_maintenance_cost

                        total_cost = total_capex + opex

                        # Calculate revenue from energy sales
                        revenue_from_sales = total_excess_energy_kwh * sell_price

                        # Update net savings by adding revenue from sales
                        net_savings = (total_consumption_lifetime * electricity_price) - (total_cost + (lifetime_uncovered_energy_kwh * electricity_price)) + revenue_from_sales

                        # Calculate TCOE based on used energy (subtract revenue)
                        tcoe = (total_cost - revenue_from_sales) / lifetime_used_energy_kwh if lifetime_used_energy_kwh > 0 else float('inf')

                        # Calculate grid energy costs without solar installation
                        grid_cost_without_solar = total_consumption_lifetime * electricity_price

                        # Calculate remaining grid energy costs with solar installation
                        grid_cost_with_solar = lifetime_uncovered_energy_kwh * electricity_price

                        # Calculate total cost with solar installation (CAPEX + OPEX + remaining grid energy costs)
                        total_cost_with_solar = total_cost + grid_cost_with_solar

                        # Recalculate net savings with revenue
                        net_savings = grid_cost_without_solar - (total_cost_with_solar - revenue_from_sales)

                        # Determine if the installation saves money
                        saves_money = net_savings > 0

                        # Store the configuration
                        fine_config = {
                            'panels_mwp': panels_mwp,
                            'batteries_mwh': batteries_mwh,
                            'portion_covered': portion_covered,
                            'portion_not_covered': 100 - portion_covered,
                            'total_cost': total_cost,
                            'total_capex': total_capex,
                            'opex': opex,
                            'project_lifetime': project_lifetime,
                            'tcoe': tcoe,
                            'net_savings': net_savings,
                            'revenue_from_sales': revenue_from_sales,
                            'saves_money': saves_money,
                            'cost_breakdown': {
                                'Panel Cost': panel_cost,
                                'Battery Cost': battery_cost,
                                'Installation Cost': installation_cost,
                                'Battery Replacement Cost': battery_replacement_cost,
                                'Panel Maintenance Cost': panel_maintenance_cost,
                                'Battery Maintenance Cost': battery_maintenance_cost,
                                'Remaining Grid Energy Cost': grid_cost_with_solar,
                                'Revenue from Energy Sales': revenue_from_sales,
                            }
                        }
                        fine_configurations.append(fine_config)

                    # Update progress bar in every iteration
                    current_iteration_fine += 1
                    progress_bar_fine.progress(min(current_iteration_fine / total_iterations_fine, 1.0))

        # Ensure the progress bar reaches 100% at the end of fine search
        progress_bar_fine.progress(1.0)

        if not fine_configurations:
            # If no configurations found in fine search, return top configurations from coarse search
            return top_configurations_coarse[:1]

        # Sort fine configurations by TCOE
        sorted_fine_configurations = sorted(fine_configurations, key=lambda x: x['tcoe'])

        # Return the best configuration
        return [sorted_fine_configurations[0]]

# Streamlit UI
st.set_page_config(page_title="Solar Plant Projection Tool", layout="wide")
st.title("🌞 Solar Plant Projection Tool")

# Sidebar Inputs
st.sidebar.header("📋 Input Parameters")

# Coordinates Input
coordinates = st.sidebar.text_input("📍 Coordinates (lat, lon)", value="48.8566, 2.3522")
try:
    latitude, longitude = map(float, coordinates.split(','))
except ValueError:
    st.sidebar.error("❗ Please enter valid coordinates separated by a comma.")
    latitude, longitude = None, None

# System Loss
system_loss = st.sidebar.number_input("⚙️ System Loss (%)", value=14.0, min_value=0.0, max_value=100.0)

# Start and End Year
start_year = st.sidebar.number_input("📅 Start Year", min_value=2005, max_value=2023, value=2020)
end_year = st.sidebar.number_input("📅 End Year", min_value=2005, max_value=2023, value=2020)

# Project Lifetime
project_lifetime = st.sidebar.number_input("⏳ Project Lifetime (years)", value=15, min_value=1)

# Panel Efficiency
wp_per_m2 = st.sidebar.number_input("🔋 Panel Efficiency (Wp/m²)", value=200, min_value=1)

# Energy Consumption Input
st.sidebar.header("🔌 Energy Consumption")
total_yearly_consumption = st.sidebar.number_input("📈 Total Yearly Consumption (kWh/year)", value=1000000, min_value=1)

# Cost Parameters
st.sidebar.header("💰 Cost Parameters")
panel_cost_per_mwp = st.sidebar.number_input("🔆 Panel Cost (€/MWp)", value=115000.0, min_value=0.0)
battery_cost_per_mwh = st.sidebar.number_input("🔋 Battery Cost (€/MWh)", value=130000.0, min_value=0.0)
installation_cost_per_mwp = st.sidebar.number_input("⚙️ Installation Cost (€/MWp)", value=800000.0, min_value=0.0)
maintenance_cost_per_mwp = st.sidebar.number_input("🛠️ Panel Maintenance Cost (€/MWp/year)", value=8000.0, min_value=0.0)
maintenance_cost_per_mwh_battery = st.sidebar.number_input("🛠️ Battery Maintenance Cost (€/MWh/year)", value=5000.0, min_value=0.0)
battery_lifespan = st.sidebar.number_input("🔋 Battery Lifespan (years)", value=15, min_value=1)
panel_degradation_rate = st.sidebar.number_input("📉 Panel Degradation Rate (% per year)", value=0.5,
                                                 min_value=0.0, max_value=100.0) / 100
min_state_of_charge = st.sidebar.number_input("🔋 Minimum State of Charge (%)", value=20.0, min_value=0.0,
                                              max_value=100.0) / 100

# Electricity Price Input
st.sidebar.header("💡 Electricity Price")
electricity_price = st.sidebar.number_input("💶 Electricity Price (€/kWh)", value=0.15, min_value=0.0)

# Sell Price Input
st.sidebar.header("💸 Energy Sales")
sell_price = st.sidebar.number_input("💶 Sell Price (€/kWh)", value=0.05, min_value=0.0, step=0.01)

# Day Start for Plotting
day_start = st.sidebar.number_input("📅 Select start day for plot (1-365)", min_value=1, max_value=365, value=1,
                                    key='day_start_selection')

# Target Coverage Slider
target_coverage = st.sidebar.slider("🎯 Target Coverage (%)", min_value=0, max_value=100, value=80, step=1)

# Calculate Button
if st.sidebar.button("✅ Calculate") and latitude is not None and longitude is not None:
    with st.spinner('Fetching data and performing optimization...'):
        result = get_equivalent_production_hours(latitude, longitude, 1000, system_loss, start_year, end_year)
        if result:
            equivalent_production_hours = result['equivalent_production_hours']
            hourly_data = result['hourly_data']

            # Convert the 'time' field to a pandas DataFrame for easier filtering
            hourly_df = pd.DataFrame(hourly_data)
            hourly_df['time'] = pd.to_datetime(hourly_df['time'], format='%Y%m%d:%H%M')

            # Generate the load curve for the entire period
            total_hours = len(hourly_data)
            load_curve = generate_fictional_load_curve(total_hours, total_yearly_consumption)

            # Initialize progress bars for coarse and fine search
            progress_bar = st.progress(0)

            # Perform optimization using the full data
            optimal_configurations = optimize_solar_battery_system(
                load_curve,
                hourly_data,
                project_lifetime,
                wp_per_m2,
                system_loss,
                start_year,
                end_year,
                target_coverage,
                panel_cost_per_mwp,
                battery_cost_per_mwh,
                installation_cost_per_mwp,
                maintenance_cost_per_mwp,
                maintenance_cost_per_mwh_battery,
                battery_lifespan,
                panel_degradation_rate,
                min_state_of_charge,
                electricity_price,
                sell_price,
                progress_bar
            )

            if optimal_configurations:
                st.success(
                    f"✅ **Optimization Complete!** Found {len(optimal_configurations)} optimal configuration(s).")
                st.write(
                    f"### 🏆 Top {len(optimal_configurations)} Configuration(s) for Location ({latitude}, {longitude})")

                for idx, config in enumerate(optimal_configurations):
                    st.subheader(f"#### Configuration {idx + 1}")
                    st.markdown(f"- **Solar Panel Capacity**: {config['panels_mwp']:.2f} MWp")
                    st.markdown(f"- **Battery Capacity**: {config['batteries_mwh']:.2f} MWh")
                    st.markdown(f"- **Portion of Consumption Covered**: {config['portion_covered']:.2f}%")
                    st.markdown(f"- **Total CAPEX**: €{config['total_capex']:,.2f}")
                    st.markdown(f"- **Total Cost of Energy (TCOE)**: €{config['tcoe']:.4f}/kWh")

                    # Display net savings or loss with revenue
                    net_savings = config['net_savings']
                    revenue_from_sales = config['revenue_from_sales']
                    if config['saves_money']:
                        st.markdown(f"- **Net Savings Over Project Lifetime**: €{net_savings:,.2f} (Including €{revenue_from_sales:,.2f} from Energy Sales)")
                        st.success("**The installation will save money over the project lifetime.**")
                    else:
                        st.markdown(f"- **Net Loss Over Project Lifetime**: €{abs(net_savings):,.2f} (Including €{revenue_from_sales:,.2f} from Energy Sales)")
                        st.warning("**The installation will not save money over the project lifetime.**")

                    st.markdown("---")

                    # Prepare data for pie chart
                    cost_breakdown = config['cost_breakdown']
                    labels = list(cost_breakdown.keys())
                    sizes = list(cost_breakdown.values())

                    # Create pie chart
                    fig, ax = plt.subplots()
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    plt.title('Cost Breakdown Over Project Lifetime')
                    st.pyplot(fig)

                # Proceed with plotting for the best configuration
                best_config = optimal_configurations[0]

                # Calculate panel area in m² using the best configuration
                panels_mwp = best_config['panels_mwp']
                batteries_mwh = best_config['batteries_mwh']
                panel_area_m2 = (panels_mwp * 1e6) / wp_per_m2  # Convert MWp to m²
                module_efficiency = wp_per_m2 / 1000  # Convert Wp/m² to kW/m²

                # Define irradiation_array
                irradiation_array = np.array([hour['G(i)'] / 1000 for hour in hourly_data])

                # Calculate solar production using the same method as in the optimization function
                solar_production = (panel_area_m2 * irradiation_array * module_efficiency) * (1 - system_loss / 100)

                # Simulate battery state over the entire period for the best configuration
                min_battery_state = batteries_mwh * 1000 * min_state_of_charge  # kWh
                max_battery_state = batteries_mwh * 1000  # kWh
                battery_state = max_battery_state  # Start at full charge
                battery_state_over_time = []
                total_excess_energy_kwh = 0  # Initialize total excess energy

                for hour in range(len(load_curve)):
                    production = solar_production[hour]  # kWh
                    consumption = load_curve[hour]

                    if production >= consumption:
                        surplus = production - consumption

                        # Charge the battery with surplus energy
                        available_storage = max_battery_state - battery_state
                        energy_to_store = min(surplus, available_storage)
                        battery_state += energy_to_store

                        # Sell excess energy to the grid if any
                        excess_energy = surplus - energy_to_store
                        if excess_energy > 0:
                            total_excess_energy_kwh += excess_energy
                    else:
                        deficit = consumption - production

                        # Discharge the battery to cover the deficit
                        available_energy = battery_state - min_battery_state
                        energy_from_battery = min(available_energy, deficit)
                        battery_state -= energy_from_battery

                    # Append current battery state to battery_state_over_time
                    battery_state_over_time.append(battery_state)

                # Calculate total revenue from energy sales
                total_revenue = total_excess_energy_kwh * sell_price

                # Select which 3 days to visualize for plotting
                start_date = pd.Timestamp(f'{start_year}-01-01') + pd.Timedelta(days=day_start - 1)
                end_date = start_date + pd.Timedelta(days=3)
                mask = (hourly_df['time'] >= start_date) & (hourly_df['time'] < end_date)
                filtered_data = hourly_df.loc[mask]

                # Slice the load_curve, solar_production, and battery_state_over_time for the selected period
                if not filtered_data.empty:
                    start_hour = filtered_data.index[0]
                    end_hour = filtered_data.index[-1] + 1  # +1 because slicing is end-exclusive

                    load_curve_filtered = load_curve[start_hour:end_hour]
                    solar_production_filtered = solar_production[start_hour:end_hour]
                    battery_state_filtered = battery_state_over_time[start_hour:end_hour]

                    # Plot the profiles
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(load_curve_filtered, label='Load (kW)', color='blue')
                    ax.plot(solar_production_filtered, label='Solar Production (kWh)', color='orange')
                    ax.plot(battery_state_filtered, label='Battery State (kWh)', color='green')
                    ax.set_xlabel('Hour')
                    ax.set_ylabel('Energy')
                    ax.set_title('Profile of Load, Solar Production, and Battery State')
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Selected period for plotting has no data.")

            else:
                st.error("❌ No optimal configuration found that meets the target coverage.")
        else:
            st.error("❌ Failed to retrieve data from PVGIS API.")
