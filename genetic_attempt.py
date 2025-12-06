"""
Usage:
    # Training and saving
    from genetic_attempt import GeneticTrainer
        trainer = GeneticTrainer(population_size=15, generations=15)
    # Loading and using a trained chromosome
    from genetic_attempt import TunableScottDickController, load_chromosome
    chromosome = load_chromosome()
    controller = TunableScottDickController(chromosome)
"""
import json
import os
import math
import numpy as np
import random
from datetime import datetime
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame import KesslerController, KesslerGame, Scenario, GraphicsType

# Chromosome len: 9 (bullet_time) + 21 (theta_delta) + 21 (ship_turn) = 51
CHROMOSOME_LENGTH = 51
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "chromosome_db.json")

# Function to save trained chromosome to database
def save_chromosome(chromosome, fitness=None, name=None, db_path=DEFAULT_DB_PATH):
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            db = json.load(f)
    else:
        db = {"chromosomes": [], "best": None}
    
    entry = {
        "chromosome": chromosome,
        "fitness": fitness,
        "name": name or f"chromosome_{len(db['chromosomes'])}",
        "timestamp": datetime.now().isoformat()
    }
    
    db["chromosomes"].append(entry)
    
    if fitness is not None:
        if db["best"] is None or fitness > db["best"].get("fitness", float('-inf')):
            db["best"] = entry
    
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)
    
    print(f"Saved chromosome to {db_path}")

# Load a chromosome from the database
# Note to Marker: import this and call this in scenario file, pass it in when initiating the class
def load_chromosome(name=None, db_path=DEFAULT_DB_PATH):
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}. Train a model first.")
        return None
    
    with open(db_path, 'r') as f:
        db = json.load(f)
    
    if name is not None:
        for entry in db["chromosomes"]:
            if entry["name"] == name:
                print(f"Loaded chromosome '{name}' (fitness: {entry.get('fitness')})")
                return entry["chromosome"]
        print(f"Chromosome '{name}' not found")
        return None
    
    if db["best"] is not None:
        print(f"Loaded best chromosome (fitness: {db['best'].get('fitness')})")
        return db["best"]["chromosome"]
    
    if db["chromosomes"]:
        entry = db["chromosomes"][-1]
        print(f"Loaded latest chromosome '{entry['name']}' (fitness: {entry.get('fitness')})")
        return entry["chromosome"]
    
    print("No chromosomes found in database")
    return None

class TunableScottDickController(KesslerController):
    """
    A fuzzy logic controller based on ThreatController with GA-tunable membership functions.
    
    The GA tunes:
    - bullet_time membership functions (S, M, L)
    - theta_delta membership functions (N_Big, N_Med, N_Fine, Z, P_Fine, P_Med, P_Big)
    - ship_turn membership functions (Hard_Left, Left, Nudge_Left, Stop, Nudge_Right, Right, Hard_Right)
    - closest_dist membership functions (Danger_Close, Near, Far)
    - dist_move membership functions (S, M, L)
    """
    
    def __init__(self, chromosome=None):
        self.eval_frames = 0
        if chromosome is None:
            chromosome = [random.uniform(0, 1) for _ in range(CHROMOSOME_LENGTH)]
        
        self.chromosome = np.asarray(chromosome, dtype=np.float64)
        self._build_fuzzy_system()

    def _get_sorted_triplet(self, start_idx, scale_min, scale_max):
        genes = np.sort(self.chromosome[start_idx:start_idx + 3])
        return genes * (scale_max - scale_min) + scale_min

    def _build_fuzzy_system(self):
        """Build the fuzzy control system matching ThreatController structure."""
        bullet_time = ctrl.Antecedent(np.arange(0, 3.0, 0.1), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.1), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        dist_move = ctrl.Consequent(np.arange(0, 2.0, 0.1), 'move_dist')
        closest_dist = ctrl.Antecedent(np.arange(0, 2000, 10), 'closest_dist')

        # Chromosome layout:
        # [0-8]: bullet_time (S, M, L) - 3 triplets
        # [9-29]: theta_delta (N_Big, N_Med, N_Fine, Z, P_Fine, P_Med, P_Big) - 7 triplets  
        # [30-50]: ship_turn (Hard_Left, Left, Nudge_Left, Stop, Nudge_Right, Right, Hard_Right) - 7 triplets
        
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, self._get_sorted_triplet(0, 0, 3.0))
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, self._get_sorted_triplet(3, 0, 3.0))
        bullet_time['L'] = fuzz.trimf(bullet_time.universe, self._get_sorted_triplet(6, 0, 3.0))
        theta_delta['N_Big'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(9, -math.pi, 0))
        theta_delta['N_Med'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(12, -math.pi, 0))
        theta_delta['N_Fine'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(15, -0.5, 0))
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(18, -0.1, 0.1))
        theta_delta['P_Fine'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(21, 0, 0.5))
        theta_delta['P_Med'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(24, 0, math.pi))
        theta_delta['P_Big'] = fuzz.trimf(theta_delta.universe, self._get_sorted_triplet(27, 0, math.pi))
        ship_turn['Hard_Left'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(30, -180, 0))
        ship_turn['Left'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(33, -180, 0))
        ship_turn['Nudge_Left'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(36, -45, 0))
        ship_turn['Stop'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(39, -10, 10))
        ship_turn['Nudge_Right'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(42, 0, 45))
        ship_turn['Right'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(45, 0, 180))
        ship_turn['Hard_Right'] = fuzz.trimf(ship_turn.universe, self._get_sorted_triplet(48, 0, 180))
        closest_dist['Danger_Close'] = fuzz.trimf(closest_dist.universe, [0, 0, 300])
        closest_dist['Near'] = fuzz.trimf(closest_dist.universe, [200, 500, 900])
        closest_dist['Far'] = fuzz.trimf(closest_dist.universe, [800, 1500, 2000])
        dist_move['S'] = fuzz.trimf(dist_move.universe, [0.1, 0.2, 0.5])
        dist_move['M'] = fuzz.trimf(dist_move.universe, [0.4, 1.0, 1.4])
        dist_move['L'] = fuzz.trimf(dist_move.universe, [1.2, 1.5, 2.0])

        self.targeting_control = ctrl.ControlSystem()
        
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Big'], ship_turn['Hard_Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Big'], ship_turn['Hard_Right']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Med'], ship_turn['Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Med'], ship_turn['Right']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Fine'] & bullet_time['S'], ship_turn['Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Fine'] & bullet_time['S'], ship_turn['Right']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Fine'] & bullet_time['L'], ship_turn['Nudge_Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Fine'] & bullet_time['L'], ship_turn['Nudge_Right']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Fine'] & bullet_time['M'], ship_turn['Nudge_Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Fine'] & bullet_time['M'], ship_turn['Nudge_Right']))
        self.targeting_control.addrule(ctrl.Rule(closest_dist['Danger_Close'], dist_move['L']))
        self.targeting_control.addrule(ctrl.Rule(closest_dist['Near'], dist_move['M']))
        self.targeting_control.addrule(ctrl.Rule(closest_dist['Far'], dist_move['S']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['Z'], ship_turn['Stop']))

    def get_intercept_solution(self, ship_pos, ship_vel, ship_heading, asteroid):
        """
        Calculates the angle error and intercept time for a specific asteroid.
        (Same as ThreatController)
        """
        ast_pos = asteroid["position"]
        ast_vel = asteroid["velocity"]
        
        rel_vx = ast_vel[0] - ship_vel[0]
        rel_vy = ast_vel[1] - ship_vel[1]
        
        dx = ship_pos[0] - ast_pos[0]
        dy = ship_pos[1] - ast_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        # Law of Cosines Setup
        theta_t = math.atan2(-dy, -dx)
        theta_v = math.atan2(rel_vy, rel_vx)
        
        theta_diff = theta_t - theta_v
        cos_theta = math.cos(theta_diff)
        
        target_vel_mag = math.sqrt(rel_vx**2 + rel_vy**2)
        bullet_speed = 800

        # Quadratic for Intercept Time
        a = target_vel_mag**2 - bullet_speed**2
        b = 2 * dist * target_vel_mag * cos_theta
        c = dist**2

        determinant = b**2 - 4*a*c
        
        bullet_t = -1
        if determinant >= 0:
            if abs(a) < 0.001:
                if b != 0:
                    bullet_t = -c / b
            else:
                t1 = (-b + math.sqrt(determinant)) / (2*a)
                t2 = (-b - math.sqrt(determinant)) / (2*a)
                times = [t for t in [t1, t2] if t >= 0]
                if times:
                    bullet_t = min(times)

        if bullet_t < 0:
            return None, None, dist
        
        # Calculate aiming point
        aim_x = ast_pos[0] + ast_vel[0] * bullet_t
        aim_y = ast_pos[1] + ast_vel[1] * bullet_t
        
        # Ship position at impact time
        my_future_x = ship_pos[0] + ship_vel[0] * bullet_t
        my_future_y = ship_pos[1] + ship_vel[1] * bullet_t
        
        # The angle we need to fire at
        aim_angle = math.atan2(aim_y - my_future_y, aim_x - my_future_x)
        
        # Difference between where we are looking and where we need to look
        current_heading_rad = math.radians(ship_heading)
        angle_error = aim_angle - current_heading_rad
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        return angle_error, bullet_t, dist

    def actions(self, ship_state, game_state):
        """
        Actions method - mirrors ThreatController exactly.
        """
        ship_pos = ship_state["position"]
        ship_vel = ship_state["velocity"]
        ship_heading = ship_state["heading"]
        ship_radius = ship_state["radius"]

        target_asteroid = None
        min_time_to_impact = float('inf')
        closest_dist_fallback = float('inf')
        fallback_asteroid = None

        for a in game_state["asteroids"]:
            dx = ship_pos[0] - a["position"][0]
            dy = ship_pos[1] - a["position"][1]
            dist = math.sqrt(dx**2 + dy**2)
            
            # Keep track of closest for fallback
            if dist < closest_dist_fallback:
                closest_dist_fallback = dist
                fallback_asteroid = a
            
            # Collision Logic
            rel_vx = a["velocity"][0] - ship_vel[0]
            rel_vy = a["velocity"][1] - ship_vel[1]
            v_dot_r = rel_vx * dx + rel_vy * dy
            v_sq = rel_vx**2 + rel_vy**2
            
            t_closest = -1
            if v_sq > 0:
                t_closest = v_dot_r / v_sq
            
            if t_closest > 0:
                future_ast_x = a["position"][0] + a["velocity"][0] * t_closest
                future_ast_y = a["position"][1] + a["velocity"][1] * t_closest
                future_ship_x = ship_pos[0] + ship_vel[0] * t_closest
                future_ship_y = ship_pos[1] + ship_vel[1] * t_closest
                
                miss_dist = math.sqrt((future_ship_x - future_ast_x)**2 + (future_ship_y - future_ast_y)**2)
                
                if miss_dist < (ship_radius + a["radius"] + 20):
                    if t_closest < min_time_to_impact:
                        min_time_to_impact = t_closest
                        target_asteroid = a

        # Select Target
        if target_asteroid is None:
            target_asteroid = fallback_asteroid
            if target_asteroid is None:
                return 0.0, 0.0, False, False

        primary_error, primary_time, _ = self.get_intercept_solution(
            ship_pos, ship_vel, ship_heading, target_asteroid
        )
        
        turn_rate = 0.0
        move_amount = 0.5
        
        if primary_error is not None:
            try:
                sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
                sim.input['bullet_time'] = min(float(primary_time), 3.0)
                sim.input['theta_delta'] = float(primary_error)
                sim.input['closest_dist'] = min(float(closest_dist_fallback), 2000)
                sim.compute()
                turn_rate = float(sim.output['ship_turn'])
                move_amount = float(sim.output['move_dist'])
            except Exception:
                turn_rate = 0.0
                move_amount = 0.5

        fire = False
        
        for a in game_state["asteroids"]:
            error, time, dist = self.get_intercept_solution(ship_pos, ship_vel, ship_heading, a)
            
            if error is None:
                continue

            # Calculate the angular size of this specific asteroid
            safe_dist = max(dist, 1.0)
            angular_radius = math.atan((a["radius"] * 0.9) / safe_dist)
            
            # If our current aim error is smaller than the size of the rock, SHOOT!
            if abs(error) <= angular_radius:
                fire = True
                break
            
            # Panic check for point-blank rocks
            if dist < (ship_radius + a["radius"] + 15):
                fire = True
                break

        # Find the closest asteroid (not necessarily the one we're shooting at)
        closest_asteroid = min(game_state["asteroids"], key=lambda a: 
            math.sqrt((a["position"][0] - ship_pos[0])**2 + (a["position"][1] - ship_pos[1])**2))
        
        closest_dist = math.sqrt(
            (closest_asteroid["position"][0] - ship_pos[0])**2 + 
            (closest_asteroid["position"][1] - ship_pos[1])**2
        )
        
        # Only thrust if asteroid is dangerously close
        danger_threshold = 200 + closest_asteroid["radius"]
        
        if closest_dist < danger_threshold:
            # Calculate angle TO the asteroid
            angle_to_asteroid = math.atan2(
                closest_asteroid["position"][1] - ship_pos[1],
                closest_asteroid["position"][0] - ship_pos[0]
            )
            
            # Ship heading in radians
            ship_heading_rad = math.radians(ship_state["heading"])
            
            # Calculate angle difference between ship heading and asteroid direction
            angle_diff = angle_to_asteroid - ship_heading_rad
            # Normalize to [-pi, pi]
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
            
            # If asteroid is roughly in front of us (within ~90 degrees), thrust backward
            # If asteroid is roughly behind us, thrust forward
            if abs(angle_diff) < math.pi / 2:
                # Asteroid is in front - thrust backward (away from it)
                thrust = float(-np.clip(move_amount * 100, 0, 100))
            else:
                # Asteroid is behind - thrust forward (away from it)
                thrust = float(np.clip(move_amount * 100, 0, 100))
        else:
            # No immediate danger, don't thrust
            thrust = 0.0
        
        drop_mine = False
        self.eval_frames += 1

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "GA Tuned Fuzzy Controller"


class GeneticTrainer:
    """
    Genetic Algorithm trainer for optimizing the fuzzy controller.
    
    Args:
        population_size: Number of individuals in each generation (default: 15)
        generations: Number of generations to evolve (default: 15)
        num_asteroids: Number of asteroids in training scenario (default: 5)
        time_limit: Time limit for each training game (default: 30)
    """
    
    def __init__(self, population_size=15, generations=15, num_asteroids=5, time_limit=30):
        self.population_size = population_size
        self.generations = generations
        self.num_asteroids = num_asteroids
        self.time_limit = time_limit
        self.best_chromosome = None
        self.best_fitness = None
        
        # Cached game settings for training (no graphics, max speed)
        self._training_settings = {
            'perf_tracker': False,
            'graphics_type': GraphicsType.NoGraphics,
            'realtime_multiplier': 0,
            'graphics_obj': None,
            'frequency': 30
        }
    
    def _create_training_scenario(self):
        """Create a training scenario for fitness evaluation."""
        return Scenario(
            name='Training',
            num_asteroids=self.num_asteroids,
            ship_states=[{'position': (400, 400), 'angle': 90, 'lives': 1, 'team': 1}],
            map_size=(1000, 800),
            time_limit=self.time_limit,
            ammo_limit_multiplier=0,
            stop_if_no_ammo=False
        )
    
    def fitness(self, chromosome):
        """
        Evaluate fitness of a chromosome by running a game simulation.
        
        Args:
            chromosome: EasyGA chromosome (list of Gene objects)
        
        Returns:
            float: Fitness score (higher is better)
        """
        # Extract values from EasyGA 'Gene' objects
        gene_values = [gene.value for gene in chromosome]
        
        # Setup the Controller with these genes
        controller = TunableScottDickController(gene_values)
        
        # Run the game
        game = KesslerGame(settings=self._training_settings)
        scenario = self._create_training_scenario()
        score, _ = game.run(scenario=scenario, controllers=[controller])
        
        # Calculate Fitness - maximize asteroids hit
        team_score = score.teams[0]
        hits = team_score.asteroids_hit
        
        # Primary goal: maximize asteroids hit
        fitness_score = hits
        return fitness_score
    
    def train(self, verbose=True):
        """
        Run genetic algorithm training.
        
        Args:
            verbose: Whether to print progress (default: True)
        
        Returns:
            list: Best chromosome values after training
        """
        import EasyGA
        
        if verbose:
            print("Initializing EasyGA...")
        
        ga = EasyGA.GA()
        ga.chromosome_length = CHROMOSOME_LENGTH
        ga.population_size = self.population_size
        ga.generation_goal = self.generations
        ga.target_fitness_type = 'max'
        ga.fitness_function_impl = self.fitness
        ga.gene_impl = lambda: random.uniform(0, 1)
        
        if verbose:
            print(f"Starting Evolution ({self.generations} generations, {self.population_size} population)...")
        
        ga.evolve()
        
        if verbose:
            ga.print_best_chromosome()
        
        # Extract best chromosome
        self.best_chromosome = [gene.value for gene in ga.population[0]]
        self.best_fitness = ga.population[0].fitness
        
        if verbose:
            print("\nTraining Complete.")
            print(f"Best Chromosome: {self.best_chromosome}")
        
        return self.best_chromosome
    
    def save(self, name=None):
        if self.best_chromosome is None:
            raise ValueError("No chromosome to save. Run train() first.")
        save_chromosome(self.best_chromosome, fitness=self.best_fitness, name=name)

if __name__ == "__main__":
    trainer = GeneticTrainer()
    best = trainer.train(verbose=True)
    trainer.save()
    print(f"\nBest chromosome saved to database.")
