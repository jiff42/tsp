## 实验1
'utlis_save_ins.py' --n_customers 15 --plane_size 200 --seed 42

```
res = ga_tsp(
        D,
        pop_size=40,
        generations=100,
        pc=0.9,
        pm=0.5,
        tournament_k=10,
        elite_two_opt=False,
        seed=2025
    )
```

```
res = sa_tsp(
        D,
        iterations=25000,
        T0=200.0,
        cooling_rate=0.9993,
        moves_per_temp=1,
        seed=2025,
        use_greedy_init=True
    )
```

```
 res = aco_tsp(
        D,
        num_ants=50,
        iterations=500,
        alpha=1.0,
        beta=5.0,
        rho=0.5,
        Q=100.0,
        use_best_so_far=True,
        best_weight=2.0,
        seed=2025
    )
```


## 实验2
'utlis_save_ins.py' --n_customers 20 --plane_size 200 --seed 42

```
res = ga_tsp(
        D,
        pop_size=80,
        generations=800,
        pc=0.9,
        pm=0.5,
        tournament_k=10,
        elite_two_opt=False,
        seed=2025
    )
```

```
res = sa_tsp(
        D,
        iterations=10000,
        T0=200.0,
        cooling_rate=0.9993,
        moves_per_temp=1,
        seed=2025,
        use_greedy_init=True
    )
```

```
res = aco_tsp(
        D,
        num_ants=80,
        iterations=1000,
        alpha=1.0,
        beta=3.0,
        rho=0.5,
        Q=100.0,
        use_best_so_far=True,
        best_weight=1.0,
        seed=2025
    )
```


## 实验3
'utlis_save_ins.py' --n_customers 40 --plane_size 200 --seed 42
```
res = ga_tsp(
        D,
        pop_size=200,
        generations=1000,
        pc=0.9,
        pm=0.5,
        tournament_k=10,
        elite_two_opt=False,
        seed=2025
    )
```

```
    res = sa_tsp(
        D,
        iterations=20000,
        T0=300.0,
        cooling_rate=0.99,
        moves_per_temp=50,
        seed=2025,
        use_greedy_init=False
    )
```

```
res = aco_tsp(
        D,
        num_ants=80,
        iterations=1000,
        alpha=1.0,
        beta=2.5,
        rho=0.5,
        Q=100.0,
        use_best_so_far=True,
        best_weight=1.0,
        seed=2025
    )
```