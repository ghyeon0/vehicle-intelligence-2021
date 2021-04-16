import numpy as np
from helpers import distance
from helpers import norm_pdf

class ParticleFilter:
    def __init__(self, num_particles):
        self.initialized = False
        self.num_particles = num_particles

    # Set the number of particles.
    # Initialize all the particles to the initial position
    #   (based on esimates of x, y, theta and their uncertainties from GPS)
    #   and all weights to 1.0.
    # Add Gaussian noise to each particle.
    def initialize(self, x, y, theta, std_x, std_y, std_theta):
        self.particles = []
        for i in range(self.num_particles):
            self.particles.append({
                'x': np.random.normal(x, std_x),
                'y': np.random.normal(y, std_y),
                't': np.random.normal(theta, std_theta),
                'w': 1.0,
                'assoc': [],
            })
        self.initialized = True

    # Add measurements to each particle and add random Gaussian noise.
    def predict(self, dt, velocity, yawrate, std_x, std_y, std_theta):
        # Be careful not to divide by zero.
        v_yr = velocity / yawrate if yawrate else 0
        yr_dt = yawrate * dt
        for p in self.particles:
            # We have to take care of very small yaw rates;
            #   apply formula for constant yaw.
            if np.fabs(yawrate) < 0.0001:
                xf = p['x'] + velocity * dt * np.cos(p['t'])
                yf = p['y'] + velocity * dt * np.sin(p['t'])
                tf = p['t']
            # Nonzero yaw rate - apply integrated formula.
            else:
                xf = p['x'] + v_yr * (np.sin(p['t'] + yr_dt) - np.sin(p['t']))
                yf = p['y'] + v_yr * (np.cos(p['t']) - np.cos(p['t'] + yr_dt))
                tf = p['t'] + yr_dt
            p['x'] = np.random.normal(xf, std_x)
            p['y'] = np.random.normal(yf, std_y)
            p['t'] = np.random.normal(tf, std_theta)

    # Find the predicted measurement that is closest to each observed
    #   measurement and assign the observed measurement to this
    #   particular landmark.
    def associate(self, predicted, observations):
        associations = []
        # For each observation, find the nearest landmark and associate it.
        #   You might want to devise and implement a more efficient algorithm.
        for o in observations:
            min_dist = -1.0
            for p in predicted:
                dist = distance(o, p)
                if min_dist < 0.0 or dist < min_dist:
                    min_dist = dist
                    min_id = p['id']
                    min_x = p['x']
                    min_y = p['y']
            association = {
                'id': min_id,
                'x': min_x,
                'y': min_y,
            }
            associations.append(association)
        # Return a list of associated landmarks that corresponds to
        #   the list of (coordinates transformed) predictions.
        return associations

    # Update the weights of each particle using a multi-variate
    #   Gaussian distribution.
    def update_weights(self, sensor_range, std_landmark_x, std_landmark_y,
                       observations, map_landmarks):
        # print(map_landmarks)
        # print(std_landmark_x, std_landmark_y)
        # TODO: For each particle, do the following:
        for particle in self.particles:
            # 1. Select the set of landmarks that are visible
            #    (within the sensor range).
            landmarks = []
            for key in map_landmarks.keys():
                dist = distance(map_landmarks[key], particle)
                if dist <= sensor_range:
                    temp_dict = map_landmarks[key]
                    temp_dict['id'] = key
                    landmarks.append(temp_dict)

            if len(landmarks) == 0: continue

            # 2. Transform each observed landmark's coordinates from the
            #    particle's coordinate system to the map's coordinates.
            p_x = particle['x']
            p_y = particle['y']
            p_t = particle['t']

            transformed_observations = []
            for observation in observations:
                m_x = p_x + observation['x'] * np.cos(p_t) - observation['y'] * np.sin(p_t)
                m_y = p_y + observation['x'] * np.sin(p_t) + observation['y'] * np.cos(p_t)
                transformed_observations.append({'x': m_x, 'y': m_y})

            # 3. Associate each transformed observation to one of the
            #    predicted (selected in Step 1) landmark positions.
            #    Use self.associate() for this purpose - it receives
            #    the predicted landmarks and observations; and returns
            #    the list of landmarks by implementing the nearest-neighbour
            #    association algorithm.
            associate_results = self.associate(landmarks, transformed_observations)

            # 4. Calculate probability of this set of observations based on
            #    a multi-variate Gaussian distribution (two variables being
            #    the x and y positions with means from associated positions
            #    and variances from std_landmark_x and std_landmark_y).
            #    The resulting probability is the product of probabilities
            #    for all the observations.
            # 5. Update the particle's weight by the calculated probability.
            particle['w'] = 1.0
            particle['assoc'] = []
            for i, associate_result in enumerate(associate_results):
                dist = distance(associate_result, particle)
                observation_distance = np.sqrt((observations[i]['x'] ** 2 + observations[i]['y'] ** 2))
                particle['w'] *= norm_pdf(dist, observation_distance, np.sqrt(std_landmark_x ** 2 + std_landmark_y ** 2)) + 1e-9
                particle['assoc'].append(associate_result['id'])

    # Resample particles with replacement with probability proportional to
    #   their weights.
    def resample(self):
        weight_list = [each['w'] for each in self.particles]
        weight_list_sum = sum(weight_list)
        for i in range(len(weight_list)):
            weight_list[i] /= weight_list_sum
    
        # TODO: Select (possibly with duplicates) the set of particles
        #       that captures the posteior belief distribution, by
        # 1. Drawing particle samples according to their weights.
            
        # 2. Make a copy of the particle; otherwise the duplicate particles
        #    will not behave independently from each other - they are
        #    references to mutable objects in Python.
        # Finally, self.particles shall contain the newly drawn set of
        #   particles.
        
        # print(weight_list)
        # print(weight_list_sum)
        new_selected_indices = np.random.choice(self.num_particles, self.num_particles, p=weight_list)
        new_selected_particles = []
        for idx in new_selected_indices:
            new_selected_particles.append(self.particles[idx].copy())
        
        self.particles = new_selected_particles
        
    # Choose the particle with the highest weight (probability)
    def get_best_particle(self):
        highest_weight = -1.0
        for p in self.particles:
            if p['w'] > highest_weight:
                highest_weight = p['w']
                best_particle = p
        return best_particle

