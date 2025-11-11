from torch import nn

env_timestep = dict({
    "ant" : 1000000,
    "hopper" : 1000000,
    "humanoid" : 1000000,
    "lunar" : 500000,
    "pendulum" : 500000,
    "reacher" : 500000,
    "walker" : 1000000
})

env_args = dict({
    "ant" : dict(
        n_envs= 1,
    ),
    "hopper" : dict(
        n_envs= 1,
    ),
    "humanoid" : dict(
        n_envs= 1,
    ),
    "lunar" : dict(
        n_envs= 1,
    ),
    "pendulum" : dict(
        n_envs= 1,
    ),
    "reacher" : dict(
        n_envs= 1,
    ),
    "walker" : dict(
        n_envs= 1,
    )
})

alg_args = dict({
    "vanilla" : dict(
        ant = dict(),
        hopper = dict(),
        humanoid = dict(),
        lunar = dict(),
        reacher = dict(),
        pendulum = dict(),
        walker = dict(),
    ),
    "caps" : dict(
        ant = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        hopper = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        humanoid = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        lunar = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.001,
            caps_lamS = 0.005,),
        reacher = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
        pendulum = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.01,
            caps_lamS = 0.05,),
        walker = dict(
            caps_sigma = 0.2,
            caps_lamT = 0.1,
            caps_lamS = 0.5,),
    ),
    "l2c2" :dict(
        ant = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        hopper = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        humanoid = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        lunar = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.001,
            l2c2_lamU = 0.1,
            l2c2_beta = 0.1),
        reacher = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
        pendulum = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.01,
            l2c2_lamU = 1.0,
            l2c2_beta = 0.1),
        walker = dict(
            l2c2_sigma = 1.0,
            l2c2_lamD = 0.1,
            l2c2_lamU = 10.0,
            l2c2_beta = 0.1),
    ),
    "asap" : dict(
        ant = dict(
            lam_p = 2.0,
            lam_s = 0.3,
            lam_t = 0.05),
        hopper = dict(
            lam_p = 2.0,
            lam_s = 0.3,
            lam_t = 0.07),
        humanoid = dict(
            lam_p = 2.0,
            lam_s = 0.3,
            lam_t = 0.05),
        lunar = dict(
            lam_p = 2.0,
            lam_s = 0.03,
            lam_t = 0.005),
        reacher = dict(
            lam_p = 2.0,
            lam_s = 0.1,
            lam_t = 0.1),
        pendulum = dict(
            lam_p = 2.0,
            lam_s = 0.03,
            lam_t = 0.005),
        walker = dict(
            lam_p = 2.0,
            lam_s = 0.3,
            lam_t = 0.05),
    ),
    "grad" : dict(
        ant = dict(
            grad_lamT = 0.1),
        hopper = dict(
            grad_lamT = 0.1),
        humanoid = dict(
            grad_lamT = 0.1),
        lunar = dict(
            grad_lamT = 0.001),
        reacher = dict(
            grad_lamT = 0.1),
        pendulum = dict(
            grad_lamT = 0.01),
        walker = dict(
            grad_lamT = 0.1),
    ),
})
