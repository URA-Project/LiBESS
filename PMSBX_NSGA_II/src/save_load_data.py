import json
from datetime import datetime
import os
import sys
from src.pmsbx_nsga.init import Chromosome, Gene, Individual, Population


class IndividualSerializer:
    @staticmethod
    def gene_to_dict(gene):
        """Convert a Gene object to a dictionary."""
        return {
            "supply_id": gene.supply_id,
            "start_date": gene.start_date.strftime("%d/%m/%Y"),
            "end_date": gene.end_date.strftime("%d/%m/%Y"),
            "scheduled_date": gene.scheduled_date.strftime("%d/%m/%Y"),
            "time_of_day": gene.time_of_day,
            "battery_type": gene.battery_type,
            "power_supply_capacity": gene.power_supply_capacity,
        }

    @staticmethod
    def chromosome_to_dict(chromosome):
        """Convert a Chromosome object to a dictionary."""
        return {
            "total_expected": chromosome.total_expected,
            "device_type": chromosome.device_type,
            "battery_type_list": chromosome.battery_type_list,
            "genes": [
                IndividualSerializer.gene_to_dict(gene) for gene in chromosome.genes
            ],
        }

    @staticmethod
    def individual_to_dict(individual):
        """Convert an Individual object to a dictionary."""
        return {
            "deadline_violation": individual.deadline_violation,
            "battery_type_violation": individual.battery_type_violation,
            "chromosomes": [
                IndividualSerializer.chromosome_to_dict(chromo)
                for chromo in individual.chromosomes
            ],
        }

    @staticmethod
    def dict_to_gene(gene_dict):
        """Convert a dictionary back to a Gene object."""
        return Gene(
            supply_id=gene_dict["supply_id"],
            start_date=datetime.strptime(gene_dict["start_date"], "%d/%m/%Y").date(),
            end_date=datetime.strptime(gene_dict["end_date"], "%d/%m/%Y").date(),
            scheduled_date=datetime.strptime(
                gene_dict["scheduled_date"], "%d/%m/%Y"
            ).date(),
            time_of_day=gene_dict["time_of_day"],
            battery_type=gene_dict["battery_type"],
            power_supply_capacity=gene_dict["power_supply_capacity"],
        )

    @staticmethod
    def dict_to_chromosome(chromosome_dict):
        """Convert a dictionary back to a Chromosome object."""
        chromosome = Chromosome(
            total_expected=chromosome_dict["total_expected"],
            device_type=chromosome_dict["device_type"],
            battery_type_list=chromosome_dict.get("battery_type_list", []),
        )
        chromosome.genes = [
            IndividualSerializer.dict_to_gene(gene_dict)
            for gene_dict in chromosome_dict["genes"]
        ]
        return chromosome

    @staticmethod
    def dict_to_individual(individual_dict):
        """Convert a dictionary back to an Individual object."""
        individual = Individual()
        individual.deadline_violation = individual_dict["deadline_violation"]
        individual.battery_type_violation = individual_dict["battery_type_violation"]
        individual.chromosomes = [
            IndividualSerializer.dict_to_chromosome(chromo_dict)
            for chromo_dict in individual_dict["chromosomes"]
        ]
        return individual

    @staticmethod
    def save_individual_to_json(individual, file_path):
        """Save an Individual object to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(
                IndividualSerializer.individual_to_dict(individual), file, indent=4
            )

    @staticmethod
    def load_individual_from_json(file_path):
        """Load an Individual object from a JSON file."""
        with open(file_path, "r") as file:
            individual_dict = json.load(file)
            return IndividualSerializer.dict_to_individual(individual_dict)


class PopulationSerializer:
    @staticmethod
    def population_to_dict(population):
        """Convert a Population object to a dictionary."""
        return {
            "individuals": [
                IndividualSerializer.individual_to_dict(individual)
                for individual in population.individuals
            ]
        }

    @staticmethod
    def dict_to_population(population_dict):
        """Convert a dictionary back to a Population object."""
        population = Population()
        population.individuals = [
            IndividualSerializer.dict_to_individual(individual_dict)
            for individual_dict in population_dict["individuals"]
        ]
        return population

    @staticmethod
    def save_population_to_json(population, file_path):
        """Save a Population object to a JSON file."""
        # Ensure the directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w") as file:
            json.dump(
                PopulationSerializer.population_to_dict(population), file, indent=4
            )

    @staticmethod
    def load_population_from_json(file_path):
        """Load a Population object from a JSON file."""
        try:
            with open(file_path, "r") as file:
                population_dict = json.load(file)
            return PopulationSerializer.dict_to_population(population_dict)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            sys.exit(1)  # Exit the program with an error code
        except json.JSONDecodeError:
            print(f"Error: The file '{file_path}' is not a valid JSON file.")
            sys.exit(1)  # Exit the program with an error code

    @staticmethod
    def save_chroms_obj_record_to_json(data, filepath):
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load_chroms_obj_record_from_json(filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Convert keys to int
            data = {int(k): v for k, v in data.items()}
            return data
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
            sys.exit(1)  # Exit the program with an error code
        except json.JSONDecodeError:
            print(f"Error: The file '{filepath}' is not a valid JSON file.")
            sys.exit(1)  # Exit the program with an error code
