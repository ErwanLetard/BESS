import requests
import streamlit as st
import numpy as np
import numpy_financial as npf  # Import numpy_financial
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

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
                                  system_loss, start_year, end_year, panel_cost_per_mwp,
                                  battery_cost_per_mwh, installation_cost_per_mwp,
                                  maintenance_cost_per_mwp,
                                  maintenance_cost_per_mwh_battery, battery_lifespan,
                                  panel_degradation_rate, min_state_of_charge, electricity_price, sell_price, progress_bar,
                                  inflation_rate, target_irr, use_batteries=True):
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
    coarse_step_size_mwp = max(max_solar_capacity_mwp / 10, 0.1)  # 10% of max capacity, ensure minimum 0.1

    if use_batteries:
        # Average Daily Consumption (kWh)
        average_daily_consumption_kwh = total_consumption_per_year / 365

        # Maximum Battery Capacity (MWh) to cover 2 days of consumption
        max_battery_capacity_mwh = (average_daily_consumption_kwh * 2) / 1000  # Convert kWh to MWh

        # Coarse Step Size for Batteries (MWh) based on total yearly consumption
        coarse_step_size_mwh = max(max_battery_capacity_mwh / 10, 0.1)  # Ensure a minimum step size

        # Generate arrays for coarse search
        panel_sizes_mwp_coarse = np.arange(0.0, max_solar_capacity_mwp + coarse_step_size_mwp, coarse_step_size_mwp)
        battery_sizes_mwh_coarse = np.arange(0.0, max_battery_capacity_mwh + coarse_step_size_mwh, coarse_step_size_mwh)

        total_iterations = len(panel_sizes_mwp_coarse) * len(battery_sizes_mwh_coarse)
    else:
        # If not using batteries, only optimize panels
        panel_sizes_mwp_coarse = np.arange(0.0, max_solar_capacity_mwp + coarse_step_size_mwp, coarse_step_size_mwp)
        battery_sizes_mwh_coarse = [0.0]  # Fixed at 0

        total_iterations = len(panel_sizes_mwp_coarse)

    current_iteration = 0

    st.write("🔍 **Phase 1: Coarse Search**")
    for panels_mwp in panel_sizes_mwp_coarse:
        panel_area_m2 = (panels_mwp * 1e6) / wp_per_m2  # Convert MWp to m²
        module_efficiency = wp_per_m2 / 1000  # Convert Wp/m² to kW/m²

        # Precompute solar production for all hours
        solar_production = (panel_area_m2 * irradiation_array * module_efficiency) * (1 - system_loss / 100)

        if use_batteries:
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

                    # Calculate grid energy costs without solar installation
                    grid_cost_without_solar = total_consumption_lifetime * electricity_price

                    # Calculate remaining grid energy costs with solar installation
                    grid_cost_with_solar = lifetime_uncovered_energy_kwh * electricity_price

                    # Calculate total cost with solar installation (CAPEX + OPEX + remaining grid energy costs)
                    total_cost_with_solar = total_cost + grid_cost_with_solar

                    # Calculate net savings by considering revenue from sales
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
                        'tcoe': (total_cost - revenue_from_sales) / lifetime_used_energy_kwh if lifetime_used_energy_kwh > 0 else float('inf'),
                        'net_savings': net_savings,
                        'revenue_from_sales': revenue_from_sales,
                        'saves_money': saves_money,
                        'lifetime_used_energy_kwh': lifetime_used_energy_kwh,
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

                    # Calculate IRR considering inflation
                    cash_flows = [-initial_capex]
                    annual_revenue = (revenue_from_sales / project_lifetime) + ((grid_cost_without_solar - grid_cost_with_solar) / project_lifetime)
                    annual_opex = opex / project_lifetime
                    replacement_cost_per_replacement = batteries_mwh * battery_cost_per_mwh

                    # Define replacement years
                    if batteries_mwh > 0 and battery_lifespan > 0:
                        replacement_years = list(range(battery_lifespan, project_lifetime + 1, battery_lifespan))
                    else:
                        replacement_years = []

                    for year in range(1, project_lifetime + 1):
                        # Adjust revenue and opex for inflation
                        revenue = annual_revenue / ((1 + inflation_rate) ** year)
                        opex_year = annual_opex / ((1 + inflation_rate) ** year)

                        # Add replacement cost if applicable
                        if year in replacement_years:
                            replacement_cost = replacement_cost_per_replacement / ((1 + inflation_rate) ** year)
                        else:
                            replacement_cost = 0.0

                        net_cash_flow = revenue - opex_year - replacement_cost
                        cash_flows.append(net_cash_flow)

                    try:
                        irr = npf.irr(cash_flows)
                        irr_percentage = irr * 100 if not np.isnan(irr) else None
                    except:
                        irr_percentage = None

                    config['irr'] = irr_percentage

                    configurations.append(config)

                # Update progress bar in every iteration
                current_iteration += 1
                progress_bar.progress(min(current_iteration / total_iterations, 1.0))
        else:
            # When not using batteries, set batteries_mwh to 0
            batteries_mwh = 0.0
            covered_by_panels = 0
            not_covered = 0

            # Initialize total excess energy for this configuration
            total_excess_energy_kwh = 0.0  # Define it here to prevent NameError

            for hour in range(len(load_curve)):
                production = solar_production[hour]
                consumption = load_curve[hour]

                if production >= consumption:
                    surplus = production - consumption

                    # Since batteries are not used, all surplus can be sold to the grid
                    total_excess_energy_kwh += surplus
                    covered_by_panels += consumption
                else:
                    deficit = consumption - production

                    # All deficit must be covered by the grid
                    not_covered += deficit
                    covered_by_panels += production

            # Calculate the portion covered
            total_consumption = sum(load_curve)
            portion_covered = (covered_by_panels / total_consumption) * 100

            # Calculate the total energy used per year
            total_used_energy_per_year = covered_by_panels / total_years

            # Adjust total used energy for the project lifetime and degradation
            degradation_factors = [(1 - panel_degradation_rate) ** year for year in range(project_lifetime)]
            lifetime_used_energy_kwh = sum(total_used_energy_per_year * df for df in degradation_factors)

            # Calculate energy not covered over the project lifetime
            lifetime_uncovered_energy_kwh = total_consumption_lifetime - lifetime_used_energy_kwh

            # Calculate initial CAPEX components
            panel_cost = panels_mwp * panel_cost_per_mwp
            battery_cost = 0.0
            installation_cost = panels_mwp * installation_cost_per_mwp

            # Total initial CAPEX
            initial_capex = panel_cost + battery_cost + installation_cost

            total_capex = initial_capex  # No battery replacement cost

            # Calculate OPEX components
            panel_maintenance_cost = maintenance_cost_per_mwp * panels_mwp * project_lifetime
            battery_maintenance_cost = 0.0  # No battery maintenance

            opex = panel_maintenance_cost + battery_maintenance_cost

            total_cost = total_capex + opex

            # Calculate revenue from energy sales
            revenue_from_sales = total_excess_energy_kwh * sell_price

            # Calculate grid energy costs without solar installation
            grid_cost_without_solar = total_consumption_lifetime * electricity_price

            # Calculate remaining grid energy costs with solar installation
            grid_cost_with_solar = lifetime_uncovered_energy_kwh * electricity_price

            # Calculate total cost with solar installation (CAPEX + OPEX + remaining grid energy costs)
            total_cost_with_solar = total_cost + grid_cost_with_solar

            # Calculate net savings by considering revenue from sales
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
                'tcoe': (total_cost - revenue_from_sales) / lifetime_used_energy_kwh if lifetime_used_energy_kwh > 0 else float('inf'),
                'net_savings': net_savings,
                'revenue_from_sales': revenue_from_sales,
                'saves_money': saves_money,
                'lifetime_used_energy_kwh': lifetime_used_energy_kwh,
                'cost_breakdown': {
                    'Panel Cost': panel_cost,
                    'Battery Cost': battery_cost,
                    'Installation Cost': installation_cost,
                    'Battery Replacement Cost': 0.0,
                    'Panel Maintenance Cost': panel_maintenance_cost,
                    'Battery Maintenance Cost': battery_maintenance_cost,
                    'Remaining Grid Energy Cost': grid_cost_with_solar,
                    'Revenue from Energy Sales': revenue_from_sales,
                }
            }

            # Calculate IRR considering inflation
            cash_flows = [-initial_capex]
            annual_revenue = (revenue_from_sales / project_lifetime) + ((grid_cost_without_solar - grid_cost_with_solar) / project_lifetime)
            annual_opex = opex / project_lifetime

            for year in range(1, project_lifetime + 1):
                # Adjust revenue and opex for inflation
                revenue = annual_revenue / ((1 + inflation_rate) ** year)
                opex_year = annual_opex / ((1 + inflation_rate) ** year)

                net_cash_flow = revenue - opex_year
                cash_flows.append(net_cash_flow)

            try:
                irr = npf.irr(cash_flows)
                irr_percentage = irr * 100 if not np.isnan(irr) else None
            except:
                irr_percentage = None

            config['irr'] = irr_percentage

            configurations.append(config)

            # Update progress bar in every iteration
            current_iteration += 1
            progress_bar.progress(min(current_iteration / total_iterations, 1.0))

    progress_bar.progress(1.0)

    # Phase 2: Fine Search
    if configurations:
        # **Filter configurations based on target IRR**
        valid_configurations = [config for config in configurations if config['irr'] is not None and not np.isnan(config['irr']) and config['irr'] >= target_irr]

        if not valid_configurations:
            st.warning("No configurations meet the target IRR in coarse search.")
            return None

        # Sort configurations by net_savings in descending order
        sorted_configurations = sorted(valid_configurations, key=lambda x: x['net_savings'], reverse=True)

        # Select the top configuration with maximum net savings
        top_configuration = sorted_configurations[0]

        st.write("🔍 **Phase 2: Fine Search**")
        fine_configurations = []

        if use_batteries:
            # Fine Step Sizes
            fine_step_size_mwp = coarse_step_size_mwp / 10
            fine_step_size_mwh = coarse_step_size_mwh / 10

            # Ensure fine step sizes are not zero
            fine_step_size_mwp = max(fine_step_size_mwp, 0.01)
            fine_step_size_mwh = max(fine_step_size_mwh, 0.01)

            # Define fine search ranges around the top configuration
            panels_mwp_center = top_configuration['panels_mwp']
            batteries_mwh_center = top_configuration['batteries_mwh']

            panels_mwp_min = max(0.0, panels_mwp_center - coarse_step_size_mwp)
            panels_mwp_max = panels_mwp_center + coarse_step_size_mwp
            battery_mwh_min = max(0.0, batteries_mwh_center - coarse_step_size_mwh)
            battery_mwh_max = batteries_mwh_center + coarse_step_size_mwh

            # Generate fine search arrays
            panel_sizes_mwp_fine = np.arange(panels_mwp_min, panels_mwp_max + fine_step_size_mwp, fine_step_size_mwp)
            battery_sizes_mwh_fine = np.arange(battery_mwh_min, battery_mwh_max + fine_step_size_mwh, fine_step_size_mwh)

            total_iterations_fine = len(panel_sizes_mwp_fine) * len(battery_sizes_mwh_fine)
        else:
            # If not using batteries, only fine-tune panels
            fine_step_size_mwp = coarse_step_size_mwp / 10
            fine_step_size_mwp = max(fine_step_size_mwp, 0.01)

            panels_mwp_center = top_configuration['panels_mwp']

            panels_mwp_min = max(0.0, panels_mwp_center - coarse_step_size_mwp)
            panels_mwp_max = panels_mwp_center + coarse_step_size_mwp

            panel_sizes_mwp_fine = np.arange(panels_mwp_min, panels_mwp_max + fine_step_size_mwp, fine_step_size_mwp)
            battery_sizes_mwh_fine = [0.0]  # Fixed at 0

            total_iterations_fine = len(panel_sizes_mwp_fine)

        current_iteration_fine = 0
        progress_bar_fine = st.progress(0)

        for panels_mwp in panel_sizes_mwp_fine:
            panel_area_m2 = (panels_mwp * 1e6) / wp_per_m2  # Convert MWp to m²
            module_efficiency = wp_per_m2 / 1000  # Convert Wp/m² to kW/m²

            # Precompute solar production for all hours
            solar_production = (panel_area_m2 * irradiation_array * module_efficiency) * (1 - system_loss / 100)

            if use_batteries:
                for batteries_mwh in battery_sizes_mwh_fine:
                    # **Fine Search Inner Loop with Batteries**

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

                        # Calculate grid energy costs without solar installation
                        grid_cost_without_solar = total_consumption_lifetime * electricity_price

                        # Calculate remaining grid energy costs with solar installation
                        grid_cost_with_solar = lifetime_uncovered_energy_kwh * electricity_price

                        # Calculate total cost with solar installation (CAPEX + OPEX + remaining grid energy costs)
                        total_cost_with_solar = total_cost + grid_cost_with_solar

                        # Calculate net savings by considering revenue from sales
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
                            'tcoe': (total_cost - revenue_from_sales) / lifetime_used_energy_kwh if lifetime_used_energy_kwh > 0 else float('inf'),
                            'net_savings': net_savings,
                            'revenue_from_sales': revenue_from_sales,
                            'saves_money': saves_money,
                            'lifetime_used_energy_kwh': lifetime_used_energy_kwh,
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

                        # Calculate IRR considering inflation
                        cash_flows = [-initial_capex]
                        annual_revenue = (revenue_from_sales / project_lifetime) + ((grid_cost_without_solar - grid_cost_with_solar) / project_lifetime)
                        annual_opex = opex / project_lifetime
                        replacement_cost_per_replacement = batteries_mwh * battery_cost_per_mwh

                        # Define replacement years
                        if batteries_mwh > 0 and battery_lifespan > 0:
                            replacement_years = list(range(battery_lifespan, project_lifetime + 1, battery_lifespan))
                        else:
                            replacement_years = []

                        for year in range(1, project_lifetime + 1):
                            # Adjust revenue and opex for inflation
                            revenue = annual_revenue / ((1 + inflation_rate) ** year)
                            opex_year = annual_opex / ((1 + inflation_rate) ** year)

                            # Add replacement cost if applicable
                            if year in replacement_years:
                                replacement_cost = replacement_cost_per_replacement / ((1 + inflation_rate) ** year)
                            else:
                                replacement_cost = 0.0

                            net_cash_flow = revenue - opex_year - replacement_cost
                            cash_flows.append(net_cash_flow)

                        try:
                            irr = npf.irr(cash_flows)
                            irr_percentage = irr * 100 if not np.isnan(irr) else None
                        except:
                            irr_percentage = None

                        config['irr'] = irr_percentage

                    if config['irr'] is not None and not np.isnan(config['irr']) and config['irr'] >= target_irr:
                        fine_configurations.append(config)

                    # Update progress bar in every iteration
                    current_iteration_fine += 1
                    progress_bar_fine.progress(min(current_iteration_fine / total_iterations_fine, 1.0))
            else:
                # **Fine Search Inner Loop without Batteries**

                # When not using batteries, set batteries_mwh to 0
                batteries_mwh = 0.0
                covered_by_panels = 0
                not_covered = 0

                # Initialize total excess energy for this configuration
                total_excess_energy_kwh = 0.0  # Define it here to prevent NameError

                for hour in range(len(load_curve)):
                    production = solar_production[hour]
                    consumption = load_curve[hour]

                    if production >= consumption:
                        surplus = production - consumption

                        # Since batteries are not used, all surplus can be sold to the grid
                        total_excess_energy_kwh += surplus
                        covered_by_panels += consumption
                    else:
                        deficit = consumption - production

                        # All deficit must be covered by the grid
                        not_covered += deficit
                        covered_by_panels += production

                # Calculate the portion covered
                total_consumption = sum(load_curve)
                portion_covered = (covered_by_panels / total_consumption) * 100

                # Calculate the total energy used per year
                total_used_energy_per_year = covered_by_panels / total_years

                # Adjust total used energy for the project lifetime and degradation
                degradation_factors = [(1 - panel_degradation_rate) ** year for year in range(project_lifetime)]
                lifetime_used_energy_kwh = sum(total_used_energy_per_year * df for df in degradation_factors)

                # Calculate energy not covered over the project lifetime
                lifetime_uncovered_energy_kwh = total_consumption_lifetime - lifetime_used_energy_kwh

                # Calculate initial CAPEX components
                panel_cost = panels_mwp * panel_cost_per_mwp
                battery_cost = 0.0
                installation_cost = panels_mwp * installation_cost_per_mwp

                # Total initial CAPEX
                initial_capex = panel_cost + battery_cost + installation_cost

                total_capex = initial_capex  # No battery replacement cost

                # Calculate OPEX components
                panel_maintenance_cost = maintenance_cost_per_mwp * panels_mwp * project_lifetime
                battery_maintenance_cost = 0.0  # No battery maintenance

                opex = panel_maintenance_cost + battery_maintenance_cost

                total_cost = total_capex + opex

                # Calculate revenue from energy sales
                revenue_from_sales = total_excess_energy_kwh * sell_price

                # Calculate grid energy costs without solar installation
                grid_cost_without_solar = total_consumption_lifetime * electricity_price

                # Calculate remaining grid energy costs with solar installation
                grid_cost_with_solar = lifetime_uncovered_energy_kwh * electricity_price

                # Calculate total cost with solar installation (CAPEX + OPEX + remaining grid energy costs)
                total_cost_with_solar = total_cost + grid_cost_with_solar

                # Calculate net savings by considering revenue from sales
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
                    'tcoe': (total_cost - revenue_from_sales) / lifetime_used_energy_kwh if lifetime_used_energy_kwh > 0 else float('inf'),
                    'net_savings': net_savings,
                    'revenue_from_sales': revenue_from_sales,
                    'saves_money': saves_money,
                    'lifetime_used_energy_kwh': lifetime_used_energy_kwh,
                    'cost_breakdown': {
                        'Panel Cost': panel_cost,
                        'Battery Cost': battery_cost,
                        'Installation Cost': installation_cost,
                        'Battery Replacement Cost': 0.0,
                        'Panel Maintenance Cost': panel_maintenance_cost,
                        'Battery Maintenance Cost': battery_maintenance_cost,
                        'Remaining Grid Energy Cost': grid_cost_with_solar,
                        'Revenue from Energy Sales': revenue_from_sales,
                    }
                }

                # Calculate IRR considering inflation
                cash_flows = [-initial_capex]
                annual_revenue = (revenue_from_sales / project_lifetime) + ((grid_cost_without_solar - grid_cost_with_solar) / project_lifetime)
                annual_opex = opex / project_lifetime

                for year in range(1, project_lifetime + 1):
                    # Adjust revenue and opex for inflation
                    revenue = annual_revenue / ((1 + inflation_rate) ** year)
                    opex_year = annual_opex / ((1 + inflation_rate) ** year)

                    net_cash_flow = revenue - opex_year
                    cash_flows.append(net_cash_flow)

                try:
                    irr = npf.irr(cash_flows)
                    irr_percentage = irr * 100 if not np.isnan(irr) else None
                except:
                    irr_percentage = None

                config['irr'] = irr_percentage

                if config['irr'] is not None and not np.isnan(config['irr']) and config['irr'] >= target_irr:
                    fine_configurations.append(config)

                # Update progress bar in every iteration
                current_iteration_fine += 1
                progress_bar_fine.progress(min(current_iteration_fine / total_iterations_fine, 1.0))

        # Ensure the progress bar reaches 100% at the end of fine search
        progress_bar_fine.progress(1.0)

        if fine_configurations:
            # Sort fine configurations by net_savings in descending order
            sorted_fine_configurations = sorted(fine_configurations, key=lambda x: x['net_savings'], reverse=True)
            # Return the best configuration
            return [sorted_fine_configurations[0]]
        else:
            st.warning("No configurations meet the target IRR in fine search.")
            # Return the top configuration from coarse search
            return [top_configuration]
    else:
        st.error("No configurations found in coarse search.")
        return None

# Streamlit UI
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

# Ensure Start Year <= End Year
if start_year > end_year:
    st.sidebar.error("❗ Start Year must be less than or equal to End Year.")

# Project Lifetime
project_lifetime = st.sidebar.number_input("⏳ Project Lifetime (years)", value=15, min_value=1)

# Panel Efficiency
wp_per_m2 = st.sidebar.number_input("🔋 Panel Efficiency (Wp/m²)", value=200, min_value=1)

# Energy Consumption Input
st.sidebar.header("🔌 Energy Consumption")

# **Load Curve Upload**
st.sidebar.subheader("📂 Upload Load Curve Excel File (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the Excel file
        load_data = pd.read_excel(uploaded_file)

        # Validate required columns
        required_columns = ['Date', 'Hour', 'Consumption']
        if not all(col in load_data.columns for col in required_columns):
            st.sidebar.error(f"❗ The uploaded file must contain the following columns: {', '.join(required_columns)}.")
            load_curve = None
            total_yearly_consumption = None
        else:
            # Convert 'Date' and 'Hour' to datetime if necessary
            load_data['Date'] = pd.to_datetime(load_data['Date'], errors='coerce')
            if load_data['Date'].isnull().any():
                st.sidebar.error("❗ Some dates in the 'Date' column could not be parsed. Please check the file format.")
                load_curve = None
                total_yearly_consumption = None
            else:
                # Ensure 'Hour' is integer and between 0-23
                if not np.issubdtype(load_data['Hour'].dtype, np.integer):
                    load_data['Hour'] = load_data['Hour'].astype(int)
                if load_data['Hour'].min() < 0 or load_data['Hour'].max() > 23:
                    st.sidebar.error("❗ The 'Hour' column must contain values between 0 and 23.")
                    load_curve = None
                    total_yearly_consumption = None
                else:
                    # Sort data by date and hour
                    load_data = load_data.sort_values(by=['Date', 'Hour'])

                    # Extract the consumption values
                    load_curve = load_data['Consumption'].tolist()

                    # Calculate total yearly consumption
                    total_yearly_consumption = sum(load_curve)

                    st.sidebar.success("✅ Load curve successfully uploaded and processed.")
    except Exception as e:
        st.sidebar.error(f"❗ An error occurred while processing the file: {e}")
        load_curve = None
        total_yearly_consumption = None
else:
    # If no file is uploaded, allow user to input total yearly consumption
    total_yearly_consumption = st.sidebar.number_input("📈 Total Yearly Consumption (kWh/year)", value=1000000, min_value=1)

# Checkbox to include batteries
include_batteries = st.sidebar.checkbox("🔋 Include Batteries in the Project", value=True)

if include_batteries:
    st.sidebar.header("🔋 Battery Parameters")
    # Cost Parameters for Batteries
    st.sidebar.subheader("💰 Cost Parameters for Batteries")
    battery_cost_per_mwh = st.sidebar.number_input("🔋 Battery Cost (€/MWh)", value=130000.0, min_value=0.0)
    maintenance_cost_per_mwh_battery = st.sidebar.number_input("🛠️ Battery Maintenance Cost (€/MWh/year)", value=5000.0, min_value=0.0)
    battery_lifespan = st.sidebar.number_input("🔋 Battery Lifespan (years)", value=15, min_value=1)
    min_state_of_charge = st.sidebar.number_input("🔋 Minimum State of Charge (%)", value=20.0, min_value=0.0,
                                                  max_value=100.0) / 100
else:
    # Set default battery-related parameters to zero or None
    battery_cost_per_mwh = 0.0
    maintenance_cost_per_mwh_battery = 0.0
    battery_lifespan = 1  # To avoid division by zero
    min_state_of_charge = 0.0

# Cost Parameters for Panels and Installation
st.sidebar.header("💰 Cost Parameters for Panels and Installation")
panel_cost_per_mwp = st.sidebar.number_input("🔆 Panel Cost (€/MWp)", value=115000.0, min_value=0.0)
installation_cost_per_mwp = st.sidebar.number_input("⚙️ Installation Cost (€/MWp)", value=800000.0, min_value=0.0)
maintenance_cost_per_mwp = st.sidebar.number_input("🛠️ Panel Maintenance Cost (€/MWp/year)", value=8000.0, min_value=0.0)

# Panel Degradation Rate
panel_degradation_rate = st.sidebar.number_input("📉 Panel Degradation Rate (% per year)", value=0.5,
                                                 min_value=0.0, max_value=100.0) / 100

# Electricity Price Input
st.sidebar.header("💡 Electricity Price")
electricity_price = st.sidebar.number_input("💶 Electricity Price (€/kWh)", value=0.15, min_value=0.0)

# Sell Price Input
st.sidebar.header("💸 Energy Sales")
sell_price = st.sidebar.number_input("💶 Sell Price (€/kWh)", value=0.05, min_value=0.0, step=0.01)

# Inflation Rate Input
st.sidebar.header("📈 Economic Parameters")
inflation_rate = st.sidebar.number_input("💹 Inflation Rate (% per year)", value=2.0, min_value=0.0, max_value=100.0, step=0.1) / 100

# **Target IRR Input**
target_irr = st.sidebar.number_input("🎯 Target Internal Rate of Return (% per year)", value=5.0, min_value=0.0, max_value=100.0, step=0.1)

# Day Start for Plotting
day_start = st.sidebar.number_input("📅 Select start day for plot (1-365)", min_value=1, max_value=365, value=1,
                                    key='day_start_selection')

# Calculate Button
if st.sidebar.button("✅ Calculate") and latitude is not None and longitude is not None:
    with st.spinner('📊 Fetching data and performing optimization...'):
        # Determine how to obtain the load curve
        if uploaded_file is not None and load_curve is not None:
            # Use the uploaded load curve
            st.write("📂 **Using Uploaded Load Curve Data**")
        else:
            # Generate a fictional load curve based on user input
            st.write("📈 **Generating Fictional Load Curve**")
            # Assuming hourly_data will be fetched after this, temporarily set hours to 8760
            load_curve = generate_fictional_load_curve(8760, total_yearly_consumption)

        # Fetch equivalent production hours
        # Note: The installed_capacity_kw parameter is not used in the current function
        result = get_equivalent_production_hours(latitude, longitude, 1000, system_loss, start_year, end_year)
        if result:
            equivalent_production_hours = result['equivalent_production_hours']
            hourly_data = result['hourly_data']

            # Convert the 'time' field to a pandas DataFrame for easier filtering
            hourly_df = pd.DataFrame(hourly_data)
            hourly_df['time'] = pd.to_datetime(hourly_df['time'], format='%Y%m%d:%H%M')

            # If load_curve was generated fictionally and depends on hourly_data length
            if uploaded_file is None or load_curve is None:
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
                progress_bar,
                inflation_rate,
                target_irr,  # Pass the target IRR
                use_batteries=include_batteries  # Pass the checkbox state
            )
            # Calculate total years based on start and end year
            total_years = end_year - start_year + 1

            # Total energy consumption per year (kWh)
            total_consumption_per_year = sum(load_curve) / total_years

            # Total energy consumption over the project lifetime
            total_consumption_lifetime = total_consumption_per_year * project_lifetime

            if optimal_configurations:
                st.success(
                    f"✅ **Optimization Complete!** Found {len(optimal_configurations)} optimal configuration(s) that meet the target IRR.")
                st.write(
                    f"### 🏆 Optimal Configuration for Location ({latitude}, {longitude})")

                for idx, config in enumerate(optimal_configurations):
                    st.subheader(f"#### Configuration {idx + 1}")
                    st.markdown(f"- **Solar Panel Capacity**: {config['panels_mwp']:.2f} MWp")
                    if include_batteries:
                        st.markdown(f"- **Battery Capacity**: {config['batteries_mwh']:.2f} MWh")
                    else:
                        st.markdown(f"- **Battery Capacity**: Not Included")
                    st.markdown(f"- **Portion of Consumption Covered**: {config['portion_covered']:.2f}%")
                    st.markdown(f"- **Total CAPEX**: €{config['total_capex']:,.2f}")
                    st.markdown(f"- **Total Cost of Energy (TCOE)**: €{config['tcoe']:.4f}/kWh")
                    if config['irr'] is not None:
                        st.markdown(f"- **Internal Rate of Return (IRR)**: {config['irr']:.2f}%")
                    else:
                        st.markdown(f"- **Internal Rate of Return (IRR)**: N/A")

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

                # Calculate total_consumption_lifetime in the main scope if not already calculated
                total_years = end_year - start_year + 1
                total_consumption_per_year = sum(load_curve) / total_years
                total_consumption_lifetime = total_consumption_per_year * project_lifetime

                # Now that best_config contains 'lifetime_used_energy_kwh', we can calculate:
                grid_cost_without_solar = total_consumption_lifetime * electricity_price
                grid_cost_with_solar = (total_consumption_lifetime - best_config['lifetime_used_energy_kwh']) * electricity_price

                # Generate annual cash flows for the best configuration
                initial_capex = best_config['total_capex'] - best_config['cost_breakdown'].get('Battery Replacement Cost', 0)
                annual_revenue = (best_config['revenue_from_sales'] / project_lifetime) + ((grid_cost_without_solar - grid_cost_with_solar) / project_lifetime)
                annual_opex = best_config['opex'] / project_lifetime
                replacement_cost_per_replacement = best_config['batteries_mwh'] * battery_cost_per_mwh

                # Define replacement years
                if best_config['batteries_mwh'] > 0 and battery_lifespan > 0:
                    replacement_years = list(range(battery_lifespan, project_lifetime, battery_lifespan))
                else:
                    replacement_years = []

                # Adjust years to go from 0 to project_lifetime - 1
                years = list(range(0, project_lifetime))

                # Prepare lists to store annual values
                revenues = []
                opex_list = []
                replacement_costs = []
                net_cash_flows = []

                for year in range(1, project_lifetime):
                    # Adjust revenue and opex for inflation
                    revenue = annual_revenue / ((1 + inflation_rate) ** year)
                    opex_year = annual_opex / ((1 + inflation_rate) ** year)

                    # Add replacement cost if applicable
                    if year in replacement_years:
                        replacement_cost = replacement_cost_per_replacement / ((1 + inflation_rate) ** year)
                    else:
                        replacement_cost = 0.0

                    net_cash_flow = revenue - opex_year - replacement_cost

                    # Append values to lists
                    revenues.append(revenue)
                    opex_list.append(-opex_year)  # Costs are negative
                    replacement_costs.append(-replacement_cost)  # Costs are negative
                    net_cash_flows.append(net_cash_flow)

                # Prepare DataFrame for plotting
                df_cash_flow = pd.DataFrame({
                    'CAPEX': [0] * len(years),
                    'Revenue': [0] * len(years),
                    'OPEX': [0] * len(years),
                    'Replacement Cost': [0] * len(years),
                    'Net Cash Flow': [0] * len(years)
                }, index=years)

                # Set values for each year using .loc
                df_cash_flow.loc[0, 'CAPEX'] = -initial_capex  # CAPEX in year 0
                df_cash_flow.loc[1:, 'Revenue'] = revenues
                df_cash_flow.loc[1:, 'OPEX'] = opex_list
                df_cash_flow.loc[1:, 'Replacement Cost'] = replacement_costs
                df_cash_flow['Net Cash Flow'] = df_cash_flow[['CAPEX', 'Revenue', 'OPEX', 'Replacement Cost']].sum(axis=1)

                # **Calculate cumulative cash flow**
                df_cash_flow['Cumulative Cash Flow'] = df_cash_flow['Net Cash Flow'].cumsum()

                # Plot the bar chart with cumulative cash flow
                fig, ax1 = plt.subplots(figsize=(12, 6))

                # Plot stacked bar chart
                df_cash_flow[['CAPEX', 'Revenue', 'OPEX', 'Replacement Cost']].plot(kind='bar', stacked=True, ax=ax1)

                # Plot net cash flow
                ax1.plot(df_cash_flow.index, df_cash_flow['Net Cash Flow'].values, color='black', marker='o', label='Net Cash Flow')

                # Create a secondary y-axis for cumulative cash flow
                ax2 = ax1.twinx()

                # Plot cumulative cash flow on the secondary y-axis
                ax2.plot(df_cash_flow.index, df_cash_flow['Cumulative Cash Flow'].values, color='green', marker='D', linestyle='--', label='Cumulative Cash Flow')

                # Set labels and titles
                ax1.set_xlabel('Year')
                ax1.set_ylabel('Cash Flow (M€)')
                ax2.set_ylabel('Cumulative Cash Flow (M€)')
                ax1.set_title('Annual Cash Flows with IRR: {:.2f}%'.format(best_config['irr']))

                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

                st.pyplot(fig)


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
                if include_batteries and batteries_mwh > 0:
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
                else:
                    # If not using batteries, set battery state to zero
                    battery_state_over_time = [0] * len(load_curve)
                    total_excess_energy_kwh = 0.0  # Initialize here to prevent NameError

                    for hour in range(len(load_curve)):
                        production = solar_production[hour]  # kWh
                        consumption = load_curve[hour]

                        if production >= consumption:
                            surplus = production - consumption

                            # Since batteries are not used, all surplus can be sold to the grid
                            total_excess_energy_kwh += surplus
                        # No battery to handle deficits

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
                    fig, ax1 = plt.subplots(figsize=(12, 6))

                    color = 'tab:blue'
                    ax1.set_xlabel('Hour')
                    ax1.set_ylabel('Load (kW)', color=color)
                    ax1.plot(load_curve_filtered, label='Load (kW)', color=color)
                    ax1.tick_params(axis='y', labelcolor=color)

                    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                    color = 'tab:orange'
                    ax2.set_ylabel('Solar Production (kWh)', color=color)  # we already handled the x-label with ax1
                    ax2.plot(solar_production_filtered, label='Solar Production (kWh)', color=color)
                    ax2.tick_params(axis='y', labelcolor=color)

                    if include_batteries and batteries_mwh > 0:
                        ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
                        ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
                        color = 'tab:green'
                        ax3.set_ylabel('Battery State (kWh)', color=color)
                        ax3.plot(battery_state_filtered, label='Battery State (kWh)', color=color)
                        ax3.tick_params(axis='y', labelcolor=color)

                    fig.tight_layout()  # otherwise the right y-label is slightly clipped
                    plt.title('Profile of Load, Solar Production, and Battery State')
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Selected period for plotting has no data.")

            else:
                st.error("❌ No optimal configuration found based on IRR.")
        else:
            st.error("❌ Failed to retrieve data from PVGIS API.")
