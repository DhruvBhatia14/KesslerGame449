# ECE 449 Intelligent Systems Engineering
from kesslergame import KesslerController
from typing import Dict, Tuple, List
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

class ThreatController(KesslerController):
    def __init__(self):
        self.eval_frames = 0

        # --- Fuzzy Variables (Same as before) ---
        bullet_time = ctrl.Antecedent(np.arange(0, 3.0, 0.1), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi, math.pi, 0.1), 'theta_delta')
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')
        dist_move = ctrl.Consequent(np.arange(0, 2.0, 0.1), 'move_dist')
        closest_dist = ctrl.Antecedent(np.arange(0, 2000, 10), 'closest_dist')

        # --- Fuzzy Sets ---
        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 1.0])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0.5, 1.5, 2.5])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 2.0, 3.0)

        theta_delta['N_Big']   = fuzz.trimf(theta_delta.universe, [-math.pi, -math.pi, -math.pi/4])
        theta_delta['N_Med']   = fuzz.trimf(theta_delta.universe, [-math.pi/2, -math.pi/4, -0.05])
        theta_delta['N_Fine']  = fuzz.trimf(theta_delta.universe, [-0.1, -0.01, 0])
        theta_delta['Z']       = fuzz.trimf(theta_delta.universe, [-0.01, 0, 0.01])
        theta_delta['P_Fine']  = fuzz.trimf(theta_delta.universe, [0, 0.01, 0.1])
        theta_delta['P_Med']   = fuzz.trimf(theta_delta.universe, [0.05, math.pi/4, math.pi/2])
        theta_delta['P_Big']   = fuzz.trimf(theta_delta.universe, [math.pi/4, math.pi, math.pi])

        ship_turn['Hard_Left']  = fuzz.trimf(ship_turn.universe, [-180, -180, -90])
        ship_turn['Left']       = fuzz.trimf(ship_turn.universe, [-90, -45, 0])
        ship_turn['Nudge_Left'] = fuzz.trimf(ship_turn.universe, [-15, -5, 0]) 
        ship_turn['Stop']       = fuzz.trimf(ship_turn.universe, [-1, 0, 1])
        ship_turn['Nudge_Right']= fuzz.trimf(ship_turn.universe, [0, 5, 15]) 
        ship_turn['Right']      = fuzz.trimf(ship_turn.universe, [0, 45, 90])
        ship_turn['Hard_Right'] = fuzz.trimf(ship_turn.universe, [90, 180, 180])

        dist_move['S'] = fuzz.trimf(dist_move.universe, [0.1, 0.2, 0.5])
        dist_move['M'] = fuzz.trimf(dist_move.universe, [0.4, 1.0, 1.4])
        dist_move['L'] = fuzz.trimf(dist_move.universe, [1.2, 1.5, 2.0])

        closest_dist['Danger_Close'] = fuzz.trimf(closest_dist.universe, [0, 0, 300])
        closest_dist['Near'] = fuzz.trimf(closest_dist.universe, [200, 500, 900])
        closest_dist['Far'] = fuzz.trimf(closest_dist.universe, [800, 1500, 2000])


        # --- Fuzzy Rules ---
        self.targeting_control = ctrl.ControlSystem()
        
        # Panic rules
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Big'], ship_turn['Hard_Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Big'], ship_turn['Hard_Right']))
        
        # Medium rules
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Med'], ship_turn['Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Med'], ship_turn['Right']))
        
        # Precision rules (Sniping)
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Fine'] & bullet_time['S'], ship_turn['Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Fine'] & bullet_time['S'], ship_turn['Right']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Fine'] & bullet_time['L'], ship_turn['Nudge_Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Fine'] & bullet_time['L'], ship_turn['Nudge_Right']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['N_Fine'] & bullet_time['M'], ship_turn['Nudge_Left']))
        self.targeting_control.addrule(ctrl.Rule(theta_delta['P_Fine'] & bullet_time['M'], ship_turn['Nudge_Right']))
        self.targeting_control.addrule(ctrl.Rule(closest_dist['Danger_Close'], dist_move['L']))
        self.targeting_control.addrule(ctrl.Rule(closest_dist['Near'], dist_move['M']))
        self.targeting_control.addrule(ctrl.Rule(closest_dist['Far'], dist_move['S']))

        # Stop
        self.targeting_control.addrule(ctrl.Rule(theta_delta['Z'], ship_turn['Stop']))

    def get_intercept_solution(self, ship_pos, ship_vel, ship_heading, asteroid):
        """
        Calculates the angle error (how far off our aim is) and intercept time
        for a specific asteroid.
        """
        ast_pos = asteroid["position"]
        ast_vel = asteroid["velocity"]
        
        # Relative velocity (Asteroid - Ship)
        # This compensates for the ship moving or drifting
        rel_vx = ast_vel[0] - ship_vel[0]
        rel_vy = ast_vel[1] - ship_vel[1]
        
        dx = ship_pos[0] - ast_pos[0]
        dy = ship_pos[1] - ast_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        
        # Law of Cosines Setup
        theta_t = math.atan2(-dy, -dx) # Angle to target (from ship)
        theta_v = math.atan2(rel_vy, rel_vx) # Angle of relative motion
        
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
                if b != 0: bullet_t = -c / b
            else:
                t1 = (-b + math.sqrt(determinant)) / (2*a)
                t2 = (-b - math.sqrt(determinant)) / (2*a)
                times = [t for t in [t1, t2] if t >= 0]
                if times:
                    bullet_t = min(times)

        if bullet_t < 0:
            return None, None, dist # No valid intercept
        
        # Calculate aiming point (Relative to Ship)
        aim_x = ast_pos[0] + ast_vel[0] * bullet_t
        aim_y = ast_pos[1] + ast_vel[1] * bullet_t
        
        # Ship position at impact time (Linear extrapolation)
        my_future_x = ship_pos[0] + ship_vel[0] * bullet_t
        my_future_y = ship_pos[1] + ship_vel[1] * bullet_t
        
        # The angle we need to fire at
        aim_angle = math.atan2(aim_y - my_future_y, aim_x - my_future_x)
        
        # Difference between where we are looking and where we need to look
        current_heading_rad = math.radians(ship_heading)
        angle_error = aim_angle - current_heading_rad
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
        
        return angle_error, bullet_t, dist

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        
        ship_pos = ship_state["position"]
        ship_vel = ship_state["velocity"]
        ship_heading = ship_state["heading"]
        ship_radius = ship_state["radius"]

        # ----------------------------------------
        # 1. THREAT DETECTION (Choose what to Look At)
        # ----------------------------------------
        target_asteroid = None
        min_time_to_impact = float('inf')
        closest_dist_fallback = float('inf')
        fallback_asteroid = None

        for a in game_state["asteroids"]:
            # Physics for Collision Detection
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
            v_dot_r = (rel_vx * dx + rel_vy * dy)
            v_sq = rel_vx**2 + rel_vy**2
            
            t_closest = -1
            if v_sq > 0: t_closest = v_dot_r / v_sq
            
            if t_closest > 0:
                # Calculate miss distance
                future_ast_x = a["position"][0] + a["velocity"][0] * t_closest
                future_ast_y = a["position"][1] + a["velocity"][1] * t_closest
                future_ship_x = ship_pos[0] + ship_vel[0] * t_closest
                future_ship_y = ship_pos[1] + ship_vel[1] * t_closest
                
                miss_dist = math.sqrt((future_ship_x - future_ast_x)**2 + (future_ship_y - future_ast_y)**2)
                
                # If it's going to hit us
                if miss_dist < (ship_radius + a["radius"] + 20):
                    if t_closest < min_time_to_impact:
                        min_time_to_impact = t_closest
                        target_asteroid = a

        # Select Target
        if target_asteroid is None:
            target_asteroid = fallback_asteroid
            if target_asteroid is None:
                return 0.0, 0.0, False, False

        # ----------------------------------------
        # 2. NAVIGATION (Turn towards Primary Target)
        # ----------------------------------------
        
        # Get solution for the primary threat to drive the Fuzzy System
        primary_error, primary_time, _ = self.get_intercept_solution(ship_pos, ship_vel, ship_heading, target_asteroid)
        
        if primary_error is not None:
            # Run Fuzzy Controller
            sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
            sim.input['bullet_time'] = min(primary_time, 3.0)
            sim.input['theta_delta'] = primary_error
            sim.input['closest_dist'] = min(closest_dist_fallback, 2000)
            try:
                sim.compute()
                turn_rate = sim.output['ship_turn']
                move_amount = sim.output['move_dist']
            except:
                turn_rate = 0
        else:
            turn_rate = 0.2

        # ----------------------------------------
        # 3. OPPORTUNITY FIRING (Check EVERYTHING)
        # ----------------------------------------
        
        fire = False
        
        # Loop through EVERY asteroid to see if we happen to be pointing at it
        for a in game_state["asteroids"]:
            error, time, dist = self.get_intercept_solution(ship_pos, ship_vel, ship_heading, a)
            
            if error is None: continue

            # Calculate the angular size of this specific asteroid
            # tan(theta) = radius / distance
            # We use 0.9 factor to be 90% sure we hit center mass
            safe_dist = max(dist, 1.0)
            angular_radius = math.atan((a["radius"] * 0.9) / safe_dist)
            
            # OPPORTUNITY CHECK:
            # If our current aim error is smaller than the size of the rock,
            # WE WILL HIT IT. SHOOT!
            if abs(error) <= angular_radius:
                fire = True
                break # We only need one reason to fire
            
            # Panic check for point-blank rocks
            if dist < (ship_radius + a["radius"] + 15):
                fire = True
                break
        # ----------------------------------------
        # 4. Distance
        # ----------------------------------------

        print(move_amount*1000)
        dx = target_asteroid["position"][0] - ship_pos[0]
        dy = target_asteroid["position"][1] - ship_pos[1]

        if dx < 15 or dy < 15:
            thrust = -np.clip(move_amount*1000, 0, 1000)
        else:
            thrust = np.clip(move_amount*1000, 0, 1000)
        drop_mine = False
        self.eval_frames += 1

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Opportunity Fire Controller"