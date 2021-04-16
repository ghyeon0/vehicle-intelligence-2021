# Week 2 - Markov Localization

---

[//]: # "Image References"
[plot]: ./markov.gif

## Assignment

You will complete the implementation of a simple Markov localizer by writing the following two functions in `markov_localizer.py`:

* `motion_model()`: For each possible prior positions, calculate the probability that the vehicle will move to the position specified by `position` given as input.
* `observation_model()`: Given the `observations`, calculate the probability of this measurement being observed using `pseudo_ranges`.

The algorithm is presented and explained in class.

All the other source files (`main.py` and `helper.py`) should be left as they are.

If you correctly implement the above functions, you expect to see a plot similar to the following:

![Expected Result of Markov Localization][plot]

If you run the program (`main.py`) without any modification to the code, it will generate only the frame of the above plot because all probabilities returned by `motion_model()` are zero by default.



## Assignment Result



![markov_result](https://user-images.githubusercontent.com/13490996/115028498-e58cef80-9eff-11eb-8105-f98a0d0fbbf2.gif)



##### motion_model 함수

```python
for i in range(map_size):
    position_prob += norm_pdf(position - i, mov, stdev) * priors[i]
```

- norm_pdf 함수의 결과와 priors 배열의 값을 이용하여 확률 계산을 수행하여 position_prob에 모두 더해서 return 하도록 코드를 작성하였다.

  

###### observation_model 함수

```python
if len(observations) == 0:
    distance_prob = 0.0
elif len(observations) > len(pseudo_ranges):
    distance_prob = 0.0
else:
    for i in range(len(observations)):
        distance_prob *= norm_pdf(observations[i], pseudo_ranges[i], stdev)
```

- observations 배열의 길이가 0이거나 pseudo_ranges의 길이보다 긴 경우는 확률을 0으로, 아닌 경우에는 observations의 확률을 모두 계산 후 곱해서 return 하도록 코드를 작성하였다. 