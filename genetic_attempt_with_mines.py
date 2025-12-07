"""
Usage:
    # Training and saving
    from genetic_attempt_with_mines import GeneticTrainer
        trainer = GeneticTrainer(population_size=15, generations=15)
        trainer.train()
        trainer.save()
    # Loading and using a trained chromosome
    from genetic_attempt_with_mines import TunableScottDickController, load_chromosome
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


def load_chromosome(name=None, db_path=DEFAULT_DB_PATH):
    if not os.path.exists(db_path):
        print(f"No database found at {db_path}. Train a model first.")
        return None

    with open(db_path, 'r') as f:
        db = json.load(f)

    if name is not None:
        for entry in db["chromosomes"]:
            if entry["name"] == name:
                print(
                    f"Loaded chromosome '{name}' (fitness: {entry.get('fitness')})")
                return entry["chromosome"]
        print(f"Chromosome '{name}' not found")
        return None

    if db["best"] is not None:
        print(f"Loaded best chromosome (fitness: {db['best'].get('fitness')})")
        return db["best"]["chromosome"]

    if db["chromosomes"]:
        entry = db["chromosomes"][-1]
        print(
            f"Loaded latest chromosome '{entry['name']}' (fitness: {entry.get('fitness')})")
        return entry["chromosome"]

    print("No chromosomes found in database")
    return None


class TunableScottDickController(KesslerController):
    def __init__(self, chromosome=None):
        self.eval_frames = 0
        if chromosome is None:
            chromosome = [random.uniform(0, 1)
                          for _ in range(CHROMOSOME_LENGTH)]

        self.chromosome = np.asarray(chromosome, dtype=np.float64)
        self._build_fuzzy_system()

    def _get_sorted_triplet(self, start_idx, scale_min, scale_max):
        genes = np.sort(self.chromosome[start_idx:start_idx + 3])
        return genes * (scale_max - scale_min) + scale_min

    def _build_fuzzy_system(self):
        bullet_time = ctrl.Antecedent(np.arange(0, 3.0, 0.1), 'bullet_time')
        theta_delta = ctrl.Antecedent(
            np.arange(-math.pi, math.pi, 0.1), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        dist_move = ctrl.Consequent(np.arange(0, 2.0, 0.1), 'move_dist')
        closest_dist = ctrl.Antecedent(np.arange(0, 2000, 10), 'closest_dist')

        bullet_time['S'] = fuzz.trimf(
            bullet_time.universe, self._get_sorted_triplet(0, 0, 3.0))
        bullet_time['M'] = fuzz.trimf(
            bullet_time.universe, self._get_sorted_triplet(3, 0, 3.0))
        bullet_time['L'] = fuzz.trimf(
            bullet_time.universe, self._get_sorted_triplet(6, 0, 3.0))

        theta_delta['N_Big'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(9, -math.pi, 0))
        theta_delta['N_Med'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(12, -math.pi, 0))
        theta_delta['N_Fine'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(15, -0.5, 0))
        theta_delta['Z'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(18, -0.1, 0.1))
        theta_delta['P_Fine'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(21, 0, 0.5))
        theta_delta['P_Med'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(24, 0, math.pi))
        theta_delta['P_Big'] = fuzz.trimf(
            theta_delta.universe, self._get_sorted_triplet(27, 0, math.pi))

        ship_turn['Hard_Left'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(30, -180, 0))
        ship_turn['Left'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(33, -180, 0))
        ship_turn['Nudge_Left'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(36, -45, 0))
        ship_turn['Stop'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(39, -10, 10))
        ship_turn['Nudge_Right'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(42, 0, 45))
        ship_turn['Right'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(45, 0, 180))
        ship_turn['Hard_Right'] = fuzz.trimf(
            ship_turn.universe, self._get_sorted_triplet(48, 0, 180))

        closest_dist['Danger_Close'] = fuzz.trimf(
            closest_dist.universe, [0, 0, 300])
        closest_dist['Near'] = fuzz.trimf(
            closest_dist.universe, [200, 500, 900])
        closest_dist['Far'] = fuzz.trimf(
            closest_dist.universe, [800, 1500, 2000])

        dist_move['S'] = fuzz.trimf(dist_move.universe, [0.1, 0.2, 0.5])
        dist_move['M'] = fuzz.trimf(dist_move.universe, [0.4, 1.0, 1.4])
        dist_move['L'] = fuzz.trimf(dist_move.universe, [1.2, 1.5, 2.0])

        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['N_Big'], ship_turn['Hard_Left']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['P_Big'], ship_turn['Hard_Right']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['N_Med'], ship_turn['Left']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['P_Med'], ship_turn['Right']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['N_Fine'] & bullet_time['S'], ship_turn['Left']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['P_Fine'] & bullet_time['S'], ship_turn['Right']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['N_Fine'] & bullet_time['L'], ship_turn['Nudge_Left']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['P_Fine'] & bullet_time['L'], ship_turn['Nudge_Right']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['N_Fine'] & bullet_time['M'], ship_turn['Nudge_Left']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['P_Fine'] & bullet_time['M'], ship_turn['Nudge_Right']))
        self.targeting_control.addrule(
            ctrl.Rule(closest_dist['Danger_Close'], dist_move['L']))
        self.targeting_control.addrule(
            ctrl.Rule(closest_dist['Near'], dist_move['M']))
        self.targeting_control.addrule(
            ctrl.Rule(closest_dist['Far'], dist_move['S']))
        self.targeting_control.addrule(
            ctrl.Rule(theta_delta['Z'], ship_turn['Stop']))

    def _get_wrapped_offset(self, pos1, pos2, map_size):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        map_w, map_h = map_size

        if dx > map_w / 2:
            dx -= map_w
        elif dx < -map_w / 2:
            dx += map_w
        if dy > map_h / 2:
            dy -= map_h
        elif dy < -map_h / 2:
            dy += map_h

        dist = math.sqrt(dx**2 + dy**2)
        return dx, dy, dist

    def _get_euclidean_offset(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dist = math.sqrt(dx**2 + dy**2)
        return dx, dy, dist

    def get_intercept_solution(self, ship_pos, ship_vel, ship_heading, asteroid, map_size):
        ast_pos = asteroid["position"]
        try:
            ast_vel = asteroid["velocity"]
        except (KeyError, AttributeError):
            ast_vel = (0, 0)

        # Use Euclidean for aiming (bullets don't wrap)
        dx, dy, dist = self._get_euclidean_offset(ship_pos, ast_pos)

        if dist > 900:
            return None, None, dist

        rel_vx = ast_vel[0] - ship_vel[0]
        rel_vy = ast_vel[1] - ship_vel[1]

        theta_t = math.atan2(dy, dx)
        theta_v = math.atan2(rel_vy, rel_vx)
        theta_diff = theta_t - theta_v
        cos_theta = math.cos(theta_diff)

        target_vel_mag = math.sqrt(rel_vx**2 + rel_vy**2)
        bullet_speed = 800

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

        aim_x = ast_pos[0] + ast_vel[0] * bullet_t
        aim_y = ast_pos[1] + ast_vel[1] * bullet_t
        my_future_x = ship_pos[0] + ship_vel[0] * bullet_t
        my_future_y = ship_pos[1] + ship_vel[1] * bullet_t

        aim_angle = math.atan2(aim_y - my_future_y, aim_x - my_future_x)
        current_heading_rad = math.radians(ship_heading)
        angle_error = aim_angle - current_heading_rad
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        return angle_error, bullet_t, dist

    def actions(self, ship_state, game_state):
        ship_pos = ship_state["position"]
        ship_vel = ship_state["velocity"]
        ship_heading = ship_state["heading"]
        ship_radius = ship_state["radius"]
        map_size = game_state["map_size"]

        try:
            mines = game_state["mines"]
        except (KeyError, TypeError, AttributeError):
            mines = []

        visible_threats = list(game_state["asteroids"]) + list(mines)

        # --- STEP 1: IDENTIFY MOST DANGEROUS OBJECT ---
        target_threat = None
        min_danger_score = float('inf')

        # We prefer hitting ANY asteroid, so we look for easiest hit
        # But we must prioritize survival if something is close.

        # Check collision threats first
        closest_collision_time = float('inf')
        collision_threat = None

        for a in visible_threats:
            dx, dy, dist = self._get_wrapped_offset(
                ship_pos, a["position"], map_size)

            try:
                obs_vel = a["velocity"]
            except (KeyError, AttributeError):
                obs_vel = (0.0, 0.0)
            try:
                obs_radius = a["radius"]
            except (KeyError, AttributeError):
                obs_radius = 20.0

            # Collision Time Calculation
            rel_vx = obs_vel[0] - ship_vel[0]
            rel_vy = obs_vel[1] - ship_vel[1]
            v_dot_r = rel_vx * dx + rel_vy * dy
            v_sq = rel_vx**2 + rel_vy**2

            t_closest = -1
            if v_sq > 0:
                t_closest = v_dot_r / v_sq

            is_collision = False
            if t_closest > 0:
                future_ast_x = ship_pos[0] + dx + obs_vel[0] * t_closest
                future_ast_y = ship_pos[1] + dy + obs_vel[1] * t_closest
                future_ship_x = ship_pos[0] + ship_vel[0] * t_closest
                future_ship_y = ship_pos[1] + ship_vel[1] * t_closest
                miss_dist = math.sqrt(
                    (future_ship_x - future_ast_x)**2 + (future_ship_y - future_ast_y)**2)

                if miss_dist < (ship_radius + obs_radius + 40):
                    is_collision = True

            # DANGER SCORE HEURISTIC
            if is_collision:
                score = t_closest
            elif v_dot_r < 0:
                closing_speed = math.sqrt(v_sq)
                t_proximity = dist / (closing_speed + 0.001)
                score = 20.0 + t_proximity
            else:
                score = 100.0 + dist

            if score < min_danger_score:
                min_danger_score = score
                target_threat = a

            # Save collision time for logic switch
            if is_collision and t_closest < closest_collision_time:
                closest_collision_time = t_closest
                collision_threat = a

        # Logic Switch:
        # 1. If we are about to die (collision < 2s), TARGET the threat to kill it.
        # 2. If we are safe, TARGET the easiest asteroid to hit to score points.

        if closest_collision_time < 2.0:
            target_threat = collision_threat
        else:
            # Find closest asteroid to SHOOT
            if game_state["asteroids"]:
                target_threat = min(game_state["asteroids"], key=lambda a:
                                    self._get_wrapped_offset(ship_pos, a["position"], map_size)[2])
            elif visible_threats:
                target_threat = visible_threats[0]

        if target_threat is None:
            return 0.0, 0.0, False, False

        # --- STEP 2: AIM ---
        primary_error, primary_time, _ = self.get_intercept_solution(
            ship_pos, ship_vel, ship_heading, target_threat, map_size
        )

        turn_rate = 0.0
        _, _, target_dist = self._get_wrapped_offset(
            ship_pos, target_threat["position"], map_size)

        if primary_error is not None:
            # Fast snap turn
            if abs(primary_error) > 0.05:
                turn_rate = 180.0 if primary_error > 0 else -180.0
            else:
                try:
                    sim = ctrl.ControlSystemSimulation(
                        self.targeting_control, flush_after_run=1)
                    sim.input['bullet_time'] = min(float(primary_time), 3.0)
                    sim.input['theta_delta'] = float(primary_error)
                    sim.input['closest_dist'] = min(float(target_dist), 2000)
                    sim.compute()
                    turn_rate = float(sim.output['ship_turn'])
                except Exception:
                    turn_rate = 0.0

        # --- STEP 3: FIRE ---
        fire = False

        # Always shoot if we have a line on ANY asteroid
        for a in game_state["asteroids"]:
            aim_error, _, hit_dist = self.get_intercept_solution(
                ship_pos, ship_vel, ship_heading, a, map_size)

            if aim_error is not None:
                safe_dist = max(hit_dist, 1.0)
                angular_radius = math.atan((a["radius"] * 0.95) / safe_dist)
                if abs(aim_error) <= angular_radius:
                    fire = True
                    break

        # Panic fire
        try:
            rad = target_threat["radius"]
        except:
            rad = 20.0
        if target_dist < (ship_radius + rad + 20):
            fire = True

        # --- STEP 4: MOVEMENT (TURRET / DEFENSE MODE) ---
        thrust = 0.0

        # Check for Mine Proximity
        closest_mine_dist = float('inf')
        closest_mine = None
        if mines:
            closest_mine = min(mines, key=lambda m:
                               self._get_wrapped_offset(ship_pos, m["position"], map_size)[2])
            _, _, closest_mine_dist = self._get_wrapped_offset(
                ship_pos, closest_mine["position"], map_size)

        # PRIORITY 1: Get away from Mines
        if closest_mine_dist < 180:
            dx, dy, _ = self._get_wrapped_offset(
                ship_pos, closest_mine["position"], map_size)
            angle_to_mine = math.atan2(dy, dx)
            ship_heading_rad = math.radians(ship_state["heading"])
            angle_diff = angle_to_mine - ship_heading_rad
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            if abs(angle_diff) < math.pi / 2:
                thrust = -300.0
            else:
                thrust = 300.0

        # PRIORITY 2: IMMINENT COLLISION AVOIDANCE
        # We use closest_collision_time calculated in Step 1.
        # If imminent impact (< 2.0s) OR super close (< 100), Dodge.
        elif closest_collision_time < 2.0 or target_dist < 80:
            dx, dy, _ = self._get_wrapped_offset(
                ship_pos, target_threat["position"], map_size)
            angle_to_obj = math.atan2(dy, dx)
            ship_heading_rad = math.radians(ship_state["heading"])
            angle_diff = angle_to_obj - ship_heading_rad
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            # Repel: If aimed at it, back up. Else, push forward.
            if abs(angle_diff) < math.pi / 2:
                thrust = -250.0  # Back up
            else:
                thrust = 250.0  # Forward

        # PRIORITY 3: BRAKES (Stabilize for shooting)
        else:
            # If we are safe, stop the ship to improve aim stability.
            current_speed = math.sqrt(ship_vel[0]**2 + ship_vel[1]**2)
            if current_speed > 5.0:
                # Calculate angle of velocity vector
                vel_angle = math.atan2(ship_vel[1], ship_vel[0])
                ship_heading_rad = math.radians(ship_state["heading"])

                # Find difference between where we are looking and where we are going
                angle_diff = vel_angle - ship_heading_rad
                angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

                # If moving forward relative to heading, reverse.
                # If moving backward relative to heading, forward.
                if abs(angle_diff) < math.pi / 2:
                    thrust = -150.0  # Brake (Reverse)
                else:
                    thrust = 150.0  # Brake (Forward)
            else:
                thrust = 0.0

        # --- STEP 5: MINE LOGIC ---
        drop_mine = False
        speed = math.sqrt(ship_vel[0]**2 + ship_vel[1]**2)

        if speed > 80 and target_dist < 150 and closest_mine_dist > 220:
            heading_rad = math.radians(ship_heading)
            ship_dir_x = math.cos(heading_rad)
            ship_dir_y = math.sin(heading_rad)
            vel_dot_heading = (
                ship_vel[0] * ship_dir_x + ship_vel[1] * ship_dir_y) / (speed + 0.0001)

            # Only drop if moving forward (don't drop while reversing!)
            if vel_dot_heading > 0.5:
                drop_mine = True

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "GA Tuned Fuzzy Controller"


class GeneticTrainer:
    """
    Genetic Algorithm trainer.
    """

    def __init__(self, population_size=15, generations=15, num_asteroids=10, time_limit=60):
        self.population_size = population_size
        self.generations = generations
        self.num_asteroids = num_asteroids
        self.time_limit = time_limit
        self.best_chromosome = None
        self.best_fitness = None

        self._training_settings = {
            'perf_tracker': False,
            'graphics_type': GraphicsType.NoGraphics,
            'realtime_multiplier': 0,
            'graphics_obj': None,
            'frequency': 30
        }

    def _create_training_scenario(self):
        # Scenario matching your test requirements
        return Scenario(
            name='Training',
            num_asteroids=self.num_asteroids,
            ship_states=[
                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1}],
            map_size=(1000, 800),
            time_limit=self.time_limit,
            ammo_limit_multiplier=0,
            stop_if_no_ammo=False
        )

    def fitness(self, chromosome):
        gene_values = [gene.value for gene in chromosome]
        controller = TunableScottDickController(gene_values)
        game = KesslerGame(settings=self._training_settings)
        scenario = self._create_training_scenario()

        # Single Player Training = FAST
        score, _ = game.run(scenario=scenario, controllers=[controller])

        team_score = score.teams[0]
        hits = team_score.asteroids_hit
        deaths = team_score.deaths
        accuracy = team_score.accuracy * 100

        # High reward for hits, moderate for accuracy, penalty for death
        fitness_score = (hits * 100) + (accuracy * 10) - (deaths * 20)
        return max(fitness_score, 0.0)

    def train(self, verbose=True):
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
            print(
                f"Starting Evolution ({self.generations} generations, {self.population_size} population)...")

        ga.evolve()

        if verbose:
            ga.print_best_chromosome()

        self.best_chromosome = [gene.value for gene in ga.population[0]]
        self.best_fitness = ga.population[0].fitness

        if verbose:
            print("\nTraining Complete.")
            print(f"Best Chromosome: {self.best_chromosome}")

        return self.best_chromosome

    def save(self, name=None):
        if self.best_chromosome is None:
            raise ValueError("No chromosome to save. Run train() first.")
        save_chromosome(self.best_chromosome,
                        fitness=self.best_fitness, name=name)


if __name__ == "__main__":
    trainer = GeneticTrainer()
    best = trainer.train(verbose=True)
    trainer.save()
    print(f"\nBest chromosome saved to database.")
