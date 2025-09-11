#!/usr/bin/env python3
"""
Example configuration for the new robust metrics fitness system
"""

# Example 1: Use preset profiles
def example_preset_profiles():
    """Show how to use preset fitness profiles"""
    
    # In your config.py or at runtime:
    from alpha_discovery.config import settings
    
    # Option 1: Use preset profile
    settings.ga.fitness_profile = "balanced"  # "safe", "balanced", "aggressive", "quality", "legacy"
    
    # Option 2: Use custom objectives
    settings.ga.objectives = ["RET_over_CVaR5", "MartinRatio", "GPR"]
    
    print("Fitness profile:", settings.ga.fitness_profile)
    print("Custom objectives:", settings.ga.objectives)

# Example 2: Runtime configuration
def example_runtime_config():
    """Show how to configure fitness at runtime"""
    
    from alpha_discovery.config import settings
    from alpha_discovery.ga.fitness_loader import get_fitness_profile
    
    # Switch between profiles easily
    profiles_to_test = ["safe", "balanced", "aggressive", "quality"]
    
    for profile in profiles_to_test:
        objectives = get_fitness_profile(profile)
        print(f"Profile '{profile}': {objectives}")

# Example 3: Custom fitness configuration
def example_custom_fitness():
    """Show how to create custom fitness configurations"""
    
    from alpha_discovery.config import settings
    
    # Custom objective combinations
    custom_configs = {
        "conservative": ["RET_over_CVaR5", "WinRate_Wilson_LB"],
        "momentum": ["GPR", "RobustSharpe_MAD"],
        "drawdown_focused": ["MartinRatio", "Calmar"],
        "comprehensive": ["RET_over_CVaR5", "MartinRatio", "GPR", "WinRate_Wilson_LB"],
    }
    
    for name, objectives in custom_configs.items():
        print(f"Custom config '{name}': {objectives}")
        # You could set this at runtime:
        # settings.ga.objectives = objectives

# Example 4: Gauntlet gates
def example_gauntlet_gates():
    """Show how to use fitness gates for filtering"""
    
    from alpha_discovery.config import settings
    
    # These gates can be used in the Gauntlet to filter setups
    gates = settings.fitness.gates
    print("Gauntlet gates:")
    for gate_name, threshold in gates.items():
        print(f"  {gate_name}: {threshold}")

if __name__ == "__main__":
    print("=== Fitness Configuration Examples ===\n")
    
    print("1. Preset Profiles:")
    example_preset_profiles()
    print()
    
    print("2. Runtime Configuration:")
    example_runtime_config()
    print()
    
    print("3. Custom Fitness:")
    example_custom_fitness()
    print()
    
    print("4. Gauntlet Gates:")
    example_gauntlet_gates()
