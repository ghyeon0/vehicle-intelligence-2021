# Week 4 - Motion Model & Particle Filters

---

[//]: # "Image References"
[empty-update]: ./empty-update.gif
[example]: ./example.gif

## Assignment

You will complete the implementation of a simple particle filter by writing the following two methods of `class ParticleFilter` defined in `particle_filter.py`:

* `update_weights()`: For each particle in the sample set, calculate the probability of the set of observations based on a multi-variate Gaussian distribution.
* `resample()`: Reconstruct the set of particles that capture the posterior belief distribution by drawing samples according to the weights.

To run the program (which generates a 2D plot), execute the following command:

```
$ python run.py
```

Without any modification to the code, you will see a resulting plot like the one below:

![Particle Filter without Proper Update & Resample][empty-update]

while a reasonable implementation of the above mentioned methods (assignments) will give you something like

![Particle Filter Example][example]

Carefully read comments in the two method bodies and write Python code that does the job.



## Assignment Result

![particle_result](https://user-images.githubusercontent.com/13490996/115034435-7666c980-9f06-11eb-94ba-c25428171524.gif)

##### update_weights 함수

```python
    def update_weights(self, sensor_range, std_landmark_x, std_landmark_y,
                       observations, map_landmarks):
        for particle in self.particles:
            landmarks = []
            for key in map_landmarks.keys():
                dist = distance(map_landmarks[key], particle)
                if dist <= sensor_range:
                    temp_dict = map_landmarks[key]
                    temp_dict['id'] = key
                    landmarks.append(temp_dict)

            if len(landmarks) == 0: continue

            p_x = particle['x']
            p_y = particle['y']
            p_t = particle['t']

            transformed_observations = []
            for observation in observations:
                m_x = p_x + observation['x'] * np.cos(p_t) - observation['y'] * np.sin(p_t)
                m_y = p_y + observation['x'] * np.sin(p_t) + observation['y'] * np.cos(p_t)
                transformed_observations.append({'x': m_x, 'y': m_y})

            associate_results = self.associate(landmarks, transformed_observations)

            particle['w'] = 1.0
            particle['assoc'] = []
            for i, associate_result in enumerate(associate_results):
                dist = distance(associate_result, particle)
                observation_distance = np.sqrt((observations[i]['x'] ** 2 + observations[i]['y'] ** 2))
                particle['w'] *= norm_pdf(dist, observation_distance, np.sqrt(std_landmark_x ** 2 + std_landmark_y ** 2)) + 1e-9
                particle['assoc'].append(associate_result['id'])
```

- 각 파티클에 대해서 보이는 landmark를 구한다.
- 그 후 좌표를 맵 좌표로 전환한다.
- associate 함수의 결과물을 이용하여 weight 및 파티클의 assoc 배열에 대해 업데이트를 진행한다. 



##### resample 함수

```python
    def resample(self):
        weight_list = [each['w'] for each in self.particles]
        weight_list_sum = sum(weight_list)
        for i in range(len(weight_list)):
            weight_list[i] /= weight_list_sum

        new_selected_indices = np.random.choice(self.num_particles, self.num_particles, p=weight_list)
        new_selected_particles = []
        for idx in new_selected_indices:
            new_selected_particles.append(self.particles[idx].copy())
        
        self.particles = new_selected_particles
```

- weight를 담은 리스트를 만들고 정규화를 진행한다. 
- 그 후 np.random.choice를 이용하여 인덱스를 가져오고, 새로운 파티클 리스트를 만들어 업데이트한다.