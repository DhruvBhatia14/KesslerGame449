import time
import math
import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from kesslergame import KesslerController, KesslerGame, Scenario, GraphicsType
import EasyGA

# =================================================================================
# 1. The Tunable Controller
#    This accepts a flat list of floats and builds the fuzzy system.
# =================================================================================


class TunableScottDickController(KesslerController):
    def __init__(self, chromosome):
        self.eval_frames = 0

        # Helper to map 0-1 genes to specific ranges (universe)
        # We assume the chromosome is a list of float values
        self.chromosome = np.array(chromosome)

        def get_sorted_triplet(start_idx, scale_min, scale_max):
            # Extract 3 genes
            genes = self.chromosome[start_idx:start_idx+3]
            # Sort them so a <= b <= c (required for trimf)
            genes = np.sort(genes)
            # Scale them from [0,1] to [scale_min, scale_max]
            rng = scale_max - scale_min
            return (genes * rng) + scale_min

        # --- Define Variables ---
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(
            np.arange(-1*math.pi, math.pi, 0.1), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        # --- Map Chromosome to Fuzzy Sets ---
        # 1. Bullet Time (0 to 1.0s) [Indices 0-8]
        bullet_time['S'] = fuzz.trimf(
            bullet_time.universe, get_sorted_triplet(0, 0, 1.0))
        bullet_time['M'] = fuzz.trimf(
            bullet_time.universe, get_sorted_triplet(3, 0, 1.0))
        bullet_time['L'] = fuzz.trimf(
            bullet_time.universe, get_sorted_triplet(6, 0, 1.0))

        # 2. Theta Delta (-PI to PI) [Indices 9-29]
        # We are defining 7 sets: NL, NM, NS, Z, PS, PM, PL
        scale_pi = math.pi
        theta_delta['NL'] = fuzz.trimf(
            theta_delta.universe, get_sorted_triplet(9, -scale_pi, scale_pi))
        theta_delta['NM'] = fuzz.trimf(
            theta_delta.universe, get_sorted_triplet(12, -scale_pi, scale_pi))
        theta_delta['NS'] = fuzz.trimf(
            theta_delta.universe, get_sorted_triplet(15, -scale_pi, scale_pi))
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, get_sorted_triplet(
            18, -scale_pi/4, scale_pi/4))  # Restrict Z to be somewhat central
        theta_delta['PS'] = fuzz.trimf(
            theta_delta.universe, get_sorted_triplet(21, -scale_pi, scale_pi))
        theta_delta['PM'] = fuzz.trimf(
            theta_delta.universe, get_sorted_triplet(24, -scale_pi, scale_pi))
        theta_delta['PL'] = fuzz.trimf(
            theta_delta.universe, get_sorted_triplet(27, -scale_pi, scale_pi))

        # 3. Ship Turn (-180 to 180 deg) [Indices 30-50]
        ship_turn['NL'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(30, -180, 180))
        ship_turn['NM'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(33, -180, 180))
        ship_turn['NS'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(36, -180, 180))
        ship_turn['Z'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(39, -180, 180))
        ship_turn['PS'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(42, -180, 180))
        ship_turn['PM'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(45, -180, 180))
        ship_turn['PL'] = fuzz.trimf(
            ship_turn.universe, get_sorted_triplet(48, -180, 180))

        # Ship Fire (Static)
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        # --- Rules ---
        # We use the standard logic, but now the shapes of 'L', 'NL', etc are determined by GA
        rules = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'],
                      (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NM'],
                      (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'],
                      (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['Z'],
                      (ship_turn['Z'],  ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'],
                      (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PM'],
                      (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'],
                      (ship_turn['PL'], ship_fire['N'])),

            ctrl.Rule(bullet_time['M'] & theta_delta['NL'],
                      (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NM'],
                      (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'],
                      (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['Z'],
                      (ship_turn['Z'],  ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'],
                      (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PM'],
                      (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'],
                      (ship_turn['PL'], ship_fire['N'])),

            ctrl.Rule(bullet_time['S'] & theta_delta['NL'],
                      (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NM'],
                      (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'],
                      (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['Z'],
                      (ship_turn['Z'],  ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'],
                      (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PM'],
                      (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'],
                      (ship_turn['PL'], ship_fire['Y'])),
        ]

        self.targeting_control = ctrl.ControlSystem(rules)

    def actions(self, ship_state, game_state):
        # ... Same Physics Math as original ScottDickController ...
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        for a in game_state["asteroids"]:
            curr_dist = math.sqrt(
                (ship_pos_x - a["position"][0])**2 + (ship_pos_y - a["position"][1])**2)
            if closest_asteroid is None or closest_asteroid["dist"] > curr_dist:
                closest_asteroid = dict(aster=a, dist=curr_dist)

        if closest_asteroid is None:
            return 0, 0, False, False

        asteroid_ship_x = ship_pos_x - closest_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["aster"]["position"][1]
        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
        asteroid_direction = math.atan2(
            closest_asteroid["aster"]["velocity"][1], closest_asteroid["aster"]["velocity"][0])
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        asteroid_vel = math.sqrt(
            closest_asteroid["aster"]["velocity"][0]**2 + closest_asteroid["aster"]["velocity"][1]**2)
        bullet_speed = 800

        targ_det = (-2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2)**2 - (
            4*(asteroid_vel**2 - bullet_speed**2) * (closest_asteroid["dist"]**2))

        if targ_det < 0:
            bullet_t = 0
        else:
            intrcpt1 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(
                targ_det)) / (2 * (asteroid_vel**2 - bullet_speed**2))
            intrcpt2 = ((2 * closest_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(
                targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
            if intrcpt1 > intrcpt2:
                bullet_t = intrcpt2 if intrcpt2 >= 0 else intrcpt1
            else:
                bullet_t = intrcpt1 if intrcpt1 >= 0 else intrcpt2

        intrcpt_x = closest_asteroid["aster"]["position"][0] + \
            closest_asteroid["aster"]["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["aster"]["position"][1] + \
            closest_asteroid["aster"]["velocity"][1] * (bullet_t+1/30)
        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),
                               (intrcpt_x - ship_pos_x))
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # --- Execute Fuzzy ---
        try:
            shooting = ctrl.ControlSystemSimulation(
                self.targeting_control, flush_after_run=1)
            shooting.input['bullet_time'] = float(bullet_t)
            shooting.input['theta_delta'] = float(shooting_theta)
            shooting.compute()
            turn_rate = shooting.output['ship_turn']
            fire = True if shooting.output['ship_fire'] >= 0 else False
        except:
            turn_rate = 0
            fire = False

        return 0, turn_rate, fire, False

    @property
    def name(self) -> str: return "EasyGA Fuzzy"

# =================================================================================
# 2. Fitness Function
#    Runs the game and calculates a score
# =================================================================================


def fitness(chromosome):
    # 1. Extract values from EasyGA 'Gene' objects
    # Similar to how the tip calculator extracted values
    gene_values = [gene.value for gene in chromosome]

    # 2. Setup the Controller with these genes
    controller = TunableScottDickController(gene_values)

    # 3. Setup the Game Environment (No Graphics for Speed!)
    scenario = Scenario(name='Training', num_asteroids=5,
                        ship_states=[
                            {'position': (400, 400), 'angle': 90, 'lives': 1, 'team': 1}],
                        map_size=(1000, 800), time_limit=30, ammo_limit_multiplier=0, stop_if_no_ammo=False)

    settings = {'perf_tracker': False,
                'graphics_type': GraphicsType.NoGraphics,  # Important for speed
                'realtime_multiplier': 0,  # Max speed
                'graphics_obj': None, 'frequency': 30}

    game = KesslerGame(settings=settings)

    # 4. Run the game
    score, _ = game.run(scenario=scenario, controllers=[controller])

    # 5. Calculate Fitness
    # Higher is better
    team_score = score.teams[0]
    hits = team_score.asteroids_hit
    deaths = team_score.deaths
    accuracy = team_score.accuracy  # 0 to 1

    # Formula: Reward hits heavily, punish death, reward accuracy
    fitness_score = (hits * 100) - (deaths * 1000) + (accuracy * 50)

    return fitness_score

# =================================================================================
# 3. EasyGA Setup and Execution
# =================================================================================


def gene_generation():
    return random.uniform(0, 1)


if __name__ == "__main__":

    print("Initializing EasyGA...")
    ga = EasyGA.GA()

    # Define Chromosome Length
    # Bullet Time (3 sets * 3 pts) + Theta (7 sets * 3 pts) + Turn (7 sets * 3 pts)
    # 9 + 21 + 21 = 51 genes
    ga.chromosome_length = 51

    # Configuration
    ga.population_size = 10     # Keep small for game simulation speed
    ga.generation_goal = 5      # Number of generations
    # We want MAX score (Tip calculator used 'min' for error)
    ga.target_fitness_type = 'max'

    # Hook up our functions
    ga.fitness_function_impl = fitness
    ga.gene_impl = lambda: gene_generation()

    print("Starting Evolution (this may take a minute)...")
    ga.evolve()

    # =================================================================================
    # 4. Results and Visualization
    # =================================================================================

    ga.print_best_chromosome()
    # Get best from population
    best_genes = [gene.value for gene in ga.population[0]]

    print("\nTraining Complete.")
    print("Running Verification Match with Graphics...")

    # Setup verification scenario
    verify_scenario = Scenario(name='Verification', num_asteroids=10,
                               ship_states=[
                                   {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1}],
                               map_size=(1000, 800), time_limit=60, ammo_limit_multiplier=0, stop_if_no_ammo=False)

    # Enable Graphics
    verify_settings = {'perf_tracker': True,
                       'graphics_type': GraphicsType.Tkinter,
                       'realtime_multiplier': 1,
                       'graphics_obj': None,
                       'frequency': 30}

    game = KesslerGame(settings=verify_settings)
    best_controller = TunableScottDickController(best_genes)

    score, perf_data = game.run(
        scenario=verify_scenario, controllers=[best_controller])

    print('Final Asteroids hit: ' +
          str([team.asteroids_hit for team in score.teams]))
    print('Final Accuracy: ' + str([team.accuracy for team in score.teams]))
