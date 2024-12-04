class GeneticParams:
    """Simple GA parameters"""
    
    def __init__(
        self,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1
    ):
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
            
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate