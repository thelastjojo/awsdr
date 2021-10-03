import math

THRESHOLD_CURVE = 0.1
CURVE_SPEED = 1.5
MAX_SPEED = 4.0


def track_direction(prev_point, next_point):
    direction = math.atan2(
        next_point[1] - prev_point[1], next_point[0] - prev_point[0])
    # Convert to degree
    direction = math.degrees(direction)
    if direction < 0:
        direction = 360 + direction
    return direction


def is_left_curve(closest_waypoints, waypoints, index=0):
    return curve_direction(closest_waypoints, waypoints, index) > THRESHOLD_CURVE


def is_right_curve(closest_waypoints, waypoints, index=0):
    return curve_direction(closest_waypoints, waypoints, index) < (THRESHOLD_CURVE * -1)


def curve_direction(closest_waypoints, waypoints, index=0):
    i_prev = closest_waypoints[0]+index
    i_next = closest_waypoints[1]+index

    if i_prev >= len(waypoints)-1:
        i_prev = i_prev - len(waypoints)
    if i_next >= len(waypoints)-1:
        i_next = i_next - len(waypoints)

    w_prev = waypoints[i_prev]
    w_next = waypoints[i_next]

    if i_next+1 >= len(waypoints)-1:
        w_next_next = waypoints[0]
    else:
        w_next_next = waypoints[i_next+1]

    current_dir = track_direction(w_prev, w_next)
    next_dir = track_direction(w_prev, w_next_next)

    curve_angle = next_dir - current_dir
    print("DR_LOG curve_angle: {0} | index: {1} | current_dir: {2} |  next_dir: {3}".format(
        curve_angle, index, current_dir, next_dir))
    return curve_angle


def calculateOptimalSpeedForStraight(closest_waypoints, waypoints, x, y):
    i = 0
    while abs(curve_direction(closest_waypoints, waypoints, i)) < THRESHOLD_CURVE:
        i += 1
    curveX = waypoints[i][0]
    curveY = waypoints[i][1]
    distance = math.sqrt((x-curveX)**2 + (y-curveY)**2)
    optimal = distance * ((MAX_SPEED - CURVE_SPEED) / 3.35) + CURVE_SPEED
    print("DR_LOG distance: {0} | optimal: {1}".format(distance, optimal))
    return optimal


def is_next_curve_left(closest_waypoints, waypoints):
    i = 0
    is_next_left = False
    is_next_right = False

    while not is_next_left and not is_next_right:
        if is_left_curve(closest_waypoints, waypoints, i):
            is_next_left = True
        if is_right_curve(closest_waypoints, waypoints, i):
            is_next_right = True
        i += 1

    return is_next_left


def reward_function(params):
    x = params['x']
    y = params['y']
    progress = params['progress']
    steps = params['steps']
    speed = params['speed']
    is_offtrack = params['is_offtrack']
    closest_waypoints = params['closest_waypoints']
    waypoints = params['waypoints']
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    is_left_of_center = params['is_left_of_center']
    steering_angle = params['steering_angle'] # negative sign (-) means right and the positive (+) sign means left

    MIN_PROGRESS_STEPS = 0.01
    MAX_PROGRESS_STEPS = 0.67  # progress/steps targets 100/150 ratio

    MAX_SPEED = 4    # from action space
    MID_SPEED = 2.67 # from action space
    MIN_SPEED = 1.33 # from action space
    NO_SPEED = 0     # from action space
    
    MAX_CURVE_SPEED = MIN_SPEED
    MIN_CURVE_SPEED = 0
    
    MAX_STRAIGHT_SPEED = MAX_SPEED
    MIN_STRAIGHT_SPEED = 0
    
    # Give a very low reward by default
    reward = 1e-3

    # normalize progress steps
    progress_steps = progress / steps
    progress_steps = (progress_steps - MIN_PROGRESS_STEPS / MAX_PROGRESS_STEPS - MIN_PROGRESS_STEPS) * 10
    
    # normalize progress
    # progress = progress / 10

    progress_steps_reward = 0
    # progress_reward = 0
    speed_reward = 0
    # track_width_reward = 0
    # track_lane_reward = 0
    steering_reward = 0

    distance_from_curve = 0
    B = MAX_SPEED

    speed_diff = 0
    speed_ratio = 0
    speed_straight_optimal = 0
    isCurve = True
    
    must_turn_left = is_left_curve(closest_waypoints, waypoints)
    must_turn_right = is_right_curve(closest_waypoints, waypoints)
    must_turn = must_turn_left or must_turn_right
    
    steering_left = steering_angle > 0
    steering_right = steering_angle < 0
    steering_straight = steering_angle == 0 # steering 0 from action space

    steering_discount = 1 # multiplier
    if must_turn:
        # B = CURVE_SPEED
        if (must_turn_left and steering_right) or (must_turn_right and steering_left):
            steering_discount = 0.75 # reduce the total reward by 25%
        
        speed_diff = abs(MAX_CURVE_SPEED - speed)
        # normalize speed in curve
        speed_ratio = 1/(0.1 + speed_diff/30)
    else:
        if not steering_straight:
            steering_discount = 0.75 # reduce the total reward by 25%
        
        speed_straight_optimal = calculateOptimalSpeedForStraight(closest_waypoints, waypoints, x, y)
        if speed_straight_optimal < 0.665:
            speed_straight_max = NO_SPEED
        elif speed_straight_optimal >= 0.665 and speed_straight_optimal < 2:
            speed_straight_max = MIN_SPEED
        elif speed_straight_optimal >= 2 and speed_straight_optimal < 3.335:
            speed_straight_max = MID_SPEED
        else
            speed_straight_max = MAX_SPEED
        
        speed_diff = abs(speed_straight_max - speed)
        # normalize speed in straight
        speed_ratio = 1/(0.1 + speed_diff/30)
        isCurve = False

        # if is_next_curve_left(closest_waypoints, waypoints) and not is_left_of_center:
        #     track_lane_reward = MAX_TRACK_LANE_REWARD
        # elif not is_next_curve_left(closest_waypoints, waypoints) and is_left_of_center:
        #     track_lane_reward = MAX_TRACK_LANE_REWARD

    # Give a reward car is on track
    if not is_offtrack:
        progress_steps_reward = math.pow(progress_steps/6.3, 3) # (progress_steps/5)**2
        # progress_reward = math.pow(progress/6.3, 3)
        speed_reward = math.pow(speed_ratio/6.3, 3) # 1/((B-speed)**2+0.25)

        reward = (progress_steps_reward + speed_reward) * steering_discount

    if reward <= 0:
        reward = 1e-3

    print('{"logtype":"DR", "progress":{0}, "steps":{1}, "speed":{2}, "x":{3}, "y":{4}, "distance_from_center":{5}, "is_left_of_center":{6}, "in_curve":{7}, "must_turn_left": {8}, "must_turn_right":{9}, "must_turn":{10}, "speed_diff":{11}, "speed_ratio":{12}, "speed_straight_optimal":{13}, "speed_straight_max":{14}, "steering_angle":{15}, "steering_left":{16}, "steering_right":{17}, "steering_straight":{18}, "steering_discount":{19}, "progress_steps_reward":{20}, "speed_reward":{21}, "total_reward":{22}, "closest_waypoints": "{23}", "w_prev":"{24}", "w_next":"{25}"}'
          .format(progress, steps, speed, x, y, distance_from_center, is_left_of_center, isCurve, must_turn_left, must_turn_right, must_turn, speed_diff, speed_ratio, speed_straight_optimal, speed_straight_max, steering_angle, steering_left, steering_right, steering_straight, steering_discount, progress_steps_reward, speed_reward, reward, closest_waypoints, waypoints[closest_waypoints[0]], waypoints[closest_waypoints[1]]))

    return float(reward)
