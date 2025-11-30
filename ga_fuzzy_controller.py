# ga_fuzzy_controller.py
#
# Fuzzy + GA-parameterised controller for Kessler Game.
# - Uses fuzzy logic to control: thrust, turn_rate, fire, drop_mine
# - Parameters (a small float vector) can be optimized by an external GA.
#
# You can:
#   - Instantiate with default parameters for a baseline agent.
#   - Or pass in a param vector from your GA in scenario_test.py.

from typing import Dict, Tuple, List
import math

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from kesslergame import KesslerController  # type: ignore


class GAFuzzyController(KesslerController):
    """
    Genetic-fuzzy controller for Kessler Game.

    Controlled outputs (per project description):
        thrust (float, m/s^2)
        turn_rate (float, deg/s)
        fire (bool)
        drop_mine (bool)  :contentReference[oaicite:2]{index=2}

    The controller’s behaviour is governed by a small parameter vector:

        idx 0: bullet_time_S_M_boundary  in [0.01, 0.20]
        idx 1: bullet_time_M_L_boundary  in [0.05, 0.60]
        idx 2: theta_small_deg           in [1.0, 6.0]
        idx 3: theta_medium_deg          in [3.0, 12.0]
        idx 4: thrust_scale              in [0.0, 400.0]
        idx 5: mine_distance_threshold   in [150.0, 500.0]

    You can evolve these params with your own GA in a separate file.
    """

    DEFAULT_PARAMS: List[float] = [
        0.05,   # bt_S_M
        0.25,   # bt_M_L
        3.0,    # theta_small_deg
        8.0,    # theta_medium_deg
        250.0,  # thrust_scale
        250.0,  # mine_distance_threshold
    ]

    def __init__(self, params: List[float] | None = None) -> None:
        self.eval_frames = 0

        if params is None:
            params = self.DEFAULT_PARAMS.copy()
        self.params = params

        # Decode list-based params so we can reuse values outside the fuzzy build
        (
            bt_S_M,
            bt_M_L,
            theta_small_deg,
            theta_medium_deg,
            thrust_scale,
            mine_dist,
        ) = self.params

        # Mine distance threshold comes from the chromosome (idx 5)
        self.mine_distance_threshold = mine_dist

        # Mine safety radius: hard-coded for now (could be another GA param later)
        self.mine_safe_radius = 120.0  # pixels

        # Track last mine we dropped
        self.last_mine_pos = None

        self.my_mines = []  # list of (x, y) where we've dropped mines

        # Now build the fuzzy systems using self.params (and fields)
        self._build_fuzzy_systems()



    # ------------------------------------------------------------------
    #  Build fuzzy controllers: targeting (turn+fire) and navigation
    #  (thrust+mine)
    # ------------------------------------------------------------------
    def _build_fuzzy_systems(self) -> None:
        bt_S_M, bt_M_L, theta_small_deg, theta_medium_deg, thrust_scale, mine_dist = self.params

        # Safety clamps so mutated / random params don't explode anything
        bt_S_M = float(min(max(bt_S_M, 0.01), 0.4))
        bt_M_L = float(min(max(bt_M_L, bt_S_M + 0.01), 0.9))

        theta_small = math.radians(max(0.5, min(theta_small_deg, 10.0)))
        theta_medium = math.radians(
            max(theta_small + math.radians(0.5), min(theta_medium_deg, 20.0))
        )

        self.thrust_scale = max(0.0, min(thrust_scale, 800.0))
        self.mine_distance_threshold = max(50.0, min(mine_dist, 600.0))

        # ================= Targeting fuzzy system ======================
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-math.pi/6, math.pi/6, 0.005),
                                      'theta_delta')

        ship_turn = ctrl.Consequent(np.arange(-180, 181, 1), 'ship_turn')
        ship_fire = ctrl.Consequent(np.arange(-1, 1.01, 0.1), 'ship_fire')

        # bullet_time fuzzy sets: S / M / L (parametrised)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0.0, 0.0, bt_S_M])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0.0, bt_S_M, bt_M_L])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, bt_S_M, bt_M_L)

        # theta_delta fuzzy sets around 0
        th_max = math.pi/6
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -th_max, -theta_medium)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe,
                                       [-th_max, -theta_medium, -theta_small])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe,
                                       [-theta_medium, -theta_small, 0.0])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe,
                                       [0.0, theta_small, theta_medium])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe,
                                       [theta_small, theta_medium, th_max])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, theta_medium, th_max)

        # Output sets for turn rate (deg/s, used by Kessler) :contentReference[oaicite:3]{index=3}
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90, -45, 0])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0, 45, 90])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        # Fire output (-1 = no fire, +1 = fire)
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        # Rules: turn toward asteroid & fire when angle is small or time is small
        rules_target = [
            # Large bullet_time → be picky about angle before firing
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),

            # Medium bullet_time → balanced behaviour
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),

            # Small bullet_time → danger / close engagement → fire aggressively
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'])),
        ]

        self._targeting_ctrl_system = ctrl.ControlSystem(rules_target)

        # ================= Navigation fuzzy system ======================
        # Controls thrust + mine drops based on distance & asteroid density.
        dist = ctrl.Antecedent(np.arange(0, 1000, 5), 'dist')
        density = ctrl.Antecedent(np.arange(0, 11, 1), 'density')

        ship_thrust = ctrl.Consequent(np.arange(0, 401, 5), 'ship_thrust')
        ship_mine = ctrl.Consequent(np.arange(-1, 1.01, 0.1), 'ship_mine')

        # Distance fuzzy sets
        dist['VERY_CLOSE'] = fuzz.zmf(dist.universe, 0, self.mine_distance_threshold * 0.6)
        dist['CLOSE'] = fuzz.trimf(dist.universe,
                                   [self.mine_distance_threshold * 0.4,
                                    self.mine_distance_threshold,
                                    self.mine_distance_threshold * 1.6])
        dist['FAR'] = fuzz.smf(dist.universe,
                               self.mine_distance_threshold,
                               min(900.0, self.mine_distance_threshold * 2.5))

        # Density (how many asteroids are near us)
        density['LOW'] = fuzz.zmf(density.universe, 0, 3)
        density['MED'] = fuzz.trimf(density.universe, [1, 4, 7])
        density['HIGH'] = fuzz.smf(density.universe, 5, 10)

        # Thrust output sets (0..thrust_scale)
        ship_thrust['ZERO'] = fuzz.trimf(ship_thrust.universe, [0, 0, 50])
        ship_thrust['LOW'] = fuzz.trimf(ship_thrust.universe,
                                        [0,
                                         self.thrust_scale * 0.3,
                                         self.thrust_scale * 0.5])
        ship_thrust['HIGH'] = fuzz.smf(ship_thrust.universe,
                                       self.thrust_scale * 0.5,
                                       self.thrust_scale)

        # Mine: -1 → no; +1 → yes
        ship_mine['N'] = fuzz.trimf(ship_mine.universe, [-1, -1, 0.0])
        ship_mine['Y'] = fuzz.trimf(ship_mine.universe, [0.0, 1, 1])

        rules_nav = [
            # Very close + high density → coast and *consider* dropping a mine.
            ctrl.Rule(dist['VERY_CLOSE'] & density['HIGH'],
                    (ship_thrust['ZERO'], ship_mine['Y'])),

            # Very close + medium density → coast, no mine (too risky / not crowded enough).
            ctrl.Rule(dist['VERY_CLOSE'] & density['MED'],
                    (ship_thrust['ZERO'], ship_mine['N'])),

            # Close + high density → some thrust (escape) and potential mine.
            ctrl.Rule(dist['CLOSE'] & density['HIGH'],
                    (ship_thrust['HIGH'], ship_mine['Y'])),

            # Close + low/med density → small thrust, no mine.
            ctrl.Rule(dist['CLOSE'] & density['LOW'],
                    (ship_thrust['LOW'], ship_mine['N'])),
            ctrl.Rule(dist['CLOSE'] & density['MED'],
                    (ship_thrust['LOW'], ship_mine['N'])),

            # Far environments → mostly conserve thrust, never mine.
            ctrl.Rule(dist['FAR'] & density['LOW'],
                    (ship_thrust['ZERO'], ship_mine['N'])),
            ctrl.Rule(dist['FAR'] & density['MED'],
                    (ship_thrust['LOW'], ship_mine['N'])),
            ctrl.Rule(dist['FAR'] & density['HIGH'],
                    (ship_thrust['LOW'], ship_mine['N'])),
        ]

        self._navigation_ctrl_system = ctrl.ControlSystem(rules_nav)

    # ------------------------------------------------------------------
    #  Helper: closest asteroid + intercept calculation (based on Dr.
    #  Dick’s guide and reference controller). :contentReference[oaicite:4]{index=4}
    # ------------------------------------------------------------------
    def _find_closest_asteroid(self, ship_state: Dict, game_state) -> Dict | None:
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]
        closest_asteroid = None

        # game_state is a GameState object that supports indexing: game_state["asteroids"]
        for a in game_state["asteroids"]:
            curr_dist = math.sqrt(
                (ship_pos_x - a["position"][0]) ** 2 +
                (ship_pos_y - a["position"][1]) ** 2
            )
            if closest_asteroid is None or curr_dist < closest_asteroid["dist"]:
                closest_asteroid = {"aster": a, "dist": curr_dist}

        return closest_asteroid


    def _compute_intercept(self, ship_state: Dict, closest_asteroid: Dict) -> Tuple[float, float]:
        """
        Compute bullet_time and theta_delta (in radians) to hit the closest asteroid,
        using the law-of-cosines based derivation in the Kessler guide. :contentReference[oaicite:5]{index=5}
        """
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        D = closest_asteroid["dist"]
        ax, ay = closest_asteroid["aster"]["position"]
        avx, avy = closest_asteroid["aster"]["velocity"]

        asteroid_ship_x = ship_pos_x - ax
        asteroid_ship_y = ship_pos_y - ay

        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
        asteroid_direction = math.atan2(avy, avx)

        theta2 = asteroid_ship_theta - asteroid_direction
        cos_theta2 = math.cos(theta2)

        asteroid_vel = math.sqrt(avx ** 2 + avy ** 2)
        bullet_speed = 800.0

        # Quadratic: a t^2 + b t + c = 0
        a = asteroid_vel ** 2 - bullet_speed ** 2
        b = 2 * D * asteroid_vel * cos_theta2
        c = D ** 2

        det = b ** 2 - 4 * a * c

        if det < 0 or abs(a) < 1e-6:
            bullet_t = 0.5  # fallback
        else:
            sqrt_det = math.sqrt(det)
            t1 = (-b + sqrt_det) / (2 * a)
            t2 = (-b - sqrt_det) / (2 * a)
            candidates = [t for t in (t1, t2) if t >= 0]
            bullet_t = min(candidates) if candidates else max(t1, t2)

        # Intercept point one tick ahead (1/30 s)
        intrcpt_x = ax + avx * (bullet_t + 1/30)
        intrcpt_y = ay + avy * (bullet_t + 1/30)

        theta1 = math.atan2(intrcpt_y - ship_pos_y, intrcpt_x - ship_pos_x)

        # Ship heading is in degrees
        shooting_theta = theta1 - (math.pi / 180.0) * ship_state["heading"]
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return bullet_t, shooting_theta

    # ------------------------------------------------------------------
    #  Main API for Kessler
    # ------------------------------------------------------------------
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Called every time step by the game engine. Returns:
            thrust, turn_rate, fire, drop_mine
        """
        self.eval_frames += 1

        closest = self._find_closest_asteroid(ship_state, game_state)
        if closest is None:
            return 0.0, 0.0, False, False

        # -------- Targeting fuzzy system (turn + fire) --------
        bullet_t, theta_delta = self._compute_intercept(ship_state, closest)

        tgt_sim = ctrl.ControlSystemSimulation(self._targeting_ctrl_system,
                                              flush_after_run=1)
        tgt_sim.input['bullet_time'] = float(max(0.0, min(bullet_t, 1.0)))
        tgt_sim.input['theta_delta'] = float(max(-math.pi/6, min(theta_delta, math.pi/6)))
        try:
            tgt_sim.compute()
            raw_turn = tgt_sim.output.get('ship_turn', 0.0)
            raw_fire = tgt_sim.output.get('ship_fire', -1.0)
        except Exception:
            raw_turn = 0.0
            raw_fire = -1.0

        try:
            turn_rate = float(raw_turn)
        except Exception:
            turn_rate = 0.0

        fire_flag = raw_fire >= 0.0


        # -------- Navigation fuzzy system (thrust + mine) -----
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        nearby_count = 0
        for a in game_state["asteroids"]:
            d = math.sqrt(
                (ship_pos_x - a["position"][0]) ** 2 +
                (ship_pos_y - a["position"][1]) ** 2
            )
            if d <= self.mine_distance_threshold * 1.5:
                nearby_count += 1


        nav_sim = ctrl.ControlSystemSimulation(self._navigation_ctrl_system,
                                           flush_after_run=1)
        nav_sim.input['dist'] = float(max(0.0, min(closest["dist"], 999.0)))
        nav_sim.input['density'] = float(max(0.0, min(nearby_count, 10)))

        # Compute nav fuzzy output with fail-safe defaults
        try:
            nav_sim.compute()
            raw_thrust = nav_sim.output.get('ship_thrust', 0.0)
            raw_mine = nav_sim.output.get('ship_mine', -1.0)
        except Exception:
            raw_thrust = 0.0
            raw_mine = -1.0

        # Base thrust from fuzzy system
        try:
            thrust_val = float(raw_thrust)
        except Exception:
            thrust_val = 0.0

        # ---------- Geometry: are we facing toward or away from the asteroid? ----------
        ax, ay = closest["aster"]["position"]
        to_asteroid_x = ax - ship_pos_x
        to_asteroid_y = ay - ship_pos_y

        heading_rad = math.radians(ship_state["heading"])
        heading_vx = math.cos(heading_rad)
        heading_vy = math.sin(heading_rad)

        # dot > 0  → asteroid somewhere in front (we're pointing toward it)
        # dot < 0  → asteroid mostly behind us (we're pointing away)
        dot = heading_vx * to_asteroid_x + heading_vy * to_asteroid_y

        # ---------- Conservative mine gating ----------
        # Only allow mines if:
        #   - fuzzy system "wants" a mine (raw_mine >= 0),
        #   - asteroid is within mine_distance_threshold,
        #   - we're facing AWAY from it (dot < 0),
        #   - and there is some crowding (nearby_count >= 2).
        mine_flag = False
        if (raw_mine >= 0.0 and
            closest["dist"] < self.mine_distance_threshold and
            dot < 0.0 and
            nearby_count >= 2):
            mine_flag = True

        if mine_flag:
            # Remember where we just dropped a mine
            self.my_mines.append(tuple(ship_state["position"]))


        # ---------- Dive-bomb avoidance for thrust ----------
        # If we're pointing toward the closest asteroid, heavily reduce thrust
        # ---------- Kite / flee logic based on distance + heading ----------
        d = closest["dist"]
        close_R1 = 250.0    # very close
        close_R2 = 500.0    # medium range

        if d < close_R1:
            # Very close: do NOT thrust toward the asteroid. Only thrust to flee.
            if dot > 0:
                # Facing it -> stop thrusting
                thrust_val = 0.0
            else:
                # Facing away -> punch it to escape
                thrust_val = max(thrust_val, self.thrust_scale * 0.7)

        elif d < close_R2:
            # Medium distance: some controlled kiting
            if dot > 0:
                # Facing it -> small thrust at most
                thrust_val = min(thrust_val, self.thrust_scale * 0.3)
            else:
                # Facing away -> moderate thrust to extend distance
                thrust_val = max(thrust_val, self.thrust_scale * 0.4)

        else:
            # Far away: don't go full send; just cruise or let inertia carry you
            thrust_val = min(thrust_val, self.thrust_scale * 0.2)

        # Final clamp
        thrust_val = max(0.0, min(thrust_val, self.thrust_scale))

        # ----------------- Build hazard avoidance vector -----------------
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        avoid_x = 0.0
        avoid_y = 0.0
        min_hazard_dist2 = float("inf")

        # 1) Asteroids as hazards
        for a in game_state["asteroids"]:
            hx, hy = a["position"]
            dx = ship_pos_x - hx
            dy = ship_pos_y - hy
            dist2 = dx*dx + dy*dy + 1e-6  # avoid division by zero
            min_hazard_dist2 = min(min_hazard_dist2, dist2)

            # Stronger repulsion when closer: weight ~ 1/dist^2
            w = 1.0 / dist2
            avoid_x += w * dx
            avoid_y += w * dy

        # 2) Our own mines as hazards (make them “scarier” with higher weight)
        for (mx, my) in self.my_mines:
            dx = ship_pos_x - mx
            dy = ship_pos_y - my
            dist2 = dx*dx + dy*dy + 1e-6
            min_hazard_dist2 = min(min_hazard_dist2, dist2)

            w = 2.0 / dist2   # mines count more than asteroids
            avoid_x += w * dx
            avoid_y += w * dy

        # ----------------- Use avoidance vector to control thrust -----------------
        # If we have a meaningful avoidance direction:
        len_avoid = math.sqrt(avoid_x*avoid_x + avoid_y*avoid_y)

        if len_avoid > 1e-3:
            # Normalized avoidance direction
            axu = avoid_x / len_avoid
            ayu = avoid_y / len_avoid

            # Current heading unit vector
            heading_rad = math.radians(ship_state["heading"])
            hx = math.cos(heading_rad)
            hy = math.sin(heading_rad)

            # cos(angle) between heading and avoidance direction:
            #   +1 = perfectly aligned with escape direction
            #    0 = perpendicular
            #   -1 = pointing straight into danger
            cos_align = hx*axu + hy*ayu

            # Distance to closest hazard (sqrt of min_hazard_dist2)
            min_hazard_dist = math.sqrt(min_hazard_dist2)

            # Tune these radii to your feel:
            R1 = 250.0   # "oh sh*t" distance
            R2 = 500.0   # moderate concern

            if min_hazard_dist < R1:
                # Very close to something bad:
                if cos_align < 0.0:
                    # We are pointing TOWARD danger -> NO thrust
                    thrust_val = 0.0
                else:
                    # Pointing away → strong thrust, at least 70% of max
                    thrust_val = max(thrust_val, self.thrust_scale * (0.7 + 0.3*cos_align))
            elif min_hazard_dist < R2:
                # Medium distance: kite
                if cos_align < 0.0:
                    # Heading is not aligned with escape; keep thrust modest
                    thrust_val = min(thrust_val, self.thrust_scale * 0.3)
                else:
                    # Good escape alignment; moderate thrust
                    thrust_val = max(thrust_val, self.thrust_scale * (0.4 + 0.3*cos_align))
            else:
                # Far from hazards: don't over-thrust; let fuzzy/thrust inertia handle it
                thrust_val = min(thrust_val, self.thrust_scale * 0.3)

        # Final clamp
        thrust_val = max(0.0, min(thrust_val, self.thrust_scale))

        # ---------------- Mine avoidance: don't hang around our own mine ----------------
        if self.last_mine_pos is not None:
            mx, my = self.last_mine_pos
            dx = mx - ship_pos_x
            dy = my - ship_pos_y
            dist_to_mine = math.sqrt(dx*dx + dy*dy)

            # Only care while we're inside some danger radius
            if dist_to_mine < self.mine_safe_radius:
                # Compute heading unit vector
                heading_rad = math.radians(ship_state["heading"])
                hx = math.cos(heading_rad)
                hy = math.sin(heading_rad)

                # Vector from ship to mine: (dx, dy)
                # dot_mine > 0  -> we're pointing toward the mine
                # dot_mine < 0  -> we're pointing away from the mine
                dot_mine = hx * dx + hy * dy

                if dot_mine > 0:
                    # Facing toward the mine -> DO NOT thrust into it
                    thrust_val = 0.0
                else:
                    # Facing away from the mine -> boost thrust to get clear
                    thrust_val = max(thrust_val, self.thrust_scale)

            # Once we're well clear, we can forget this mine (it probably exploded or is irrelevant)
            if dist_to_mine > self.mine_safe_radius * 1.5:
                self.last_mine_pos = None


        return thrust_val, turn_rate, bool(fire_flag), bool(mine_flag)

    @property
    def name(self) -> str:  # type: ignore[override]
        return "GAFuzzyController"
