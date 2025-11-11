# Smoothness Baseline

## 환경 설정 (setting test environment)
conda를 설치한 후 아래 명령어를 통해 env를 설치 및 실행합니다.
After installing conda, install and run env using the following commands.
```
cd smoothness_baseline
conda env create -f gym.yml
conda activate gym
```

## testcode

smoothness_baseline/ppo/test/test_all.py 파일을 다음과 같이 실행합니다.
Run the smoothens_baseline/ppo/test/test_all.py file as follows.
```
cd smoothness_baseline
python ppo/tests/test_all.py --envs pendulum humanoid ant reacher hopper lunar
```

--envs 파라미터로 실행시킬 환경을 설정할 수 있습니다.
You can set the environment to run with the --envs parameter.
모든 알고리즘을 테스트 하는것이 기본 설정이며 필요에 따라 아래와 같이 --algs 옵션으로 테스트할 알고리즘을 선택할 수 있습니다.
Testing all algorithms is the default setting, and you can choose which algorithms to test with the --algs option as below as needed.
```
    python ppo/tests/test_all.py --algs asap
```

## 하이퍼 파라미터 변경 (change hyper parameter)

알고리즘의 하이퍼 파라미터는 smoothness_baseline/ppo/tests/modules/params.py 파일을 수정하여 변경할 수 있습니다.
The hyperparameters of the algorithm can be changed by modifying the smoothens_baseline/ppo/tests/modules/params.py file.
```
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
```

## 결과 출력 (print result)

결과는 results/pths 폴더에 자동으로 출력됩니다.
The results are automatically output to the results/pths folder.
